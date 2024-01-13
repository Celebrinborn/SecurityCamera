import os
import pandas as pd
from typing import Dict, Optional, Tuple, Any
from pathlib import Path
import pyodbc
import datetime
from camera.frame import Frame
import logging
import time
from uuid import UUID
import inspect

import threading

import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, VARBINARY, UniqueConstraint, TypeDecorator
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import expression
from sqlalchemy.dialects.mssql import UNIQUEIDENTIFIER
from urllib.parse import quote_plus
import uuid
logger = logging.getLogger(__name__)


Base = declarative_base()

class GUID(TypeDecorator):
    """Platform-independent GUID type.
    
    Uses string on all platforms, with optional native UUID support.
    """
    impl = UNIQUEIDENTIFIER

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'mssql':
            return str(value)
        else:
            raise ValueError("Unsupported database dialect")

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        else:
            return uuid.UUID(value)
class SQL_Video(Base):
    __tablename__ = 'videos'
    __table_args__ = {'schema': 'cameras'}

    video_file_name = Column(String(255), primary_key=True, nullable=False)
    created_at = Column(DateTime, nullable=False)
    retain_until = Column(DateTime, nullable=False)
    deleted_at = Column(DateTime, nullable=False)
class SQL_Frame(Base):
    __tablename__ = 'frames'
    __table_args__ = {'schema': 'cameras'}

    frame_counter_int = Column(Integer, nullable=False)
    frame_guid = Column(GUID(), primary_key=True, default=uuid.uuid4)
    video_file_name = Column(String(255), ForeignKey('cameras.videos.video_file_name'), nullable=False)
class SQL_Screenshot(Base):
    __tablename__ = 'screenshots'
    __table_args__ = {'schema': 'cameras'}

    screenshot_id = Column(Integer, primary_key=True)
    camera_id = Column(Integer, ForeignKey('cameras.cameras.camera_id'), nullable=False)
    taken_at = Column(DateTime, nullable=False)
    screenshot_image = Column(VARBINARY, nullable=False)
class SQL_Camera(Base):
    __tablename__ = 'cameras'
    __table_args__ = {'schema': 'cameras'}

    camera_id = Column(Integer, primary_key=True)
    camera_name = Column(String(255), nullable=False)
class SQL_MotionDetected(Base):
    __tablename__ = 'motion_detected'
    __table_args__ = {'schema': 'cameras'}

    motion_detected_id = Column(Integer, primary_key=True)
    motion_amount = Column(Integer, nullable=False)
    frame_guid = Column(String(36), ForeignKey('cameras.frames.frame_guid'), nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=expression.func.getdate())
class SQL_MotionQueue(Base):
    __tablename__ = 'motion_queue'
    __table_args__ = {'schema': 'cameras'}

    motion_queue_id = Column(Integer, primary_key=True)
    motion_amount = Column(Integer, nullable=False)
    frame_guid = Column(String(36), nullable=False)
    created_at = Column(DateTime, nullable=False, server_default=expression.func.getdate())


class SQLManager:
    _instance = None  # Class-level variable to hold the singleton instance

    _lock:threading.Lock = threading.Lock()
    _conn:pyodbc.Connection
    _connection_string:str

    _engine: sqlalchemy.engine.Engine

    def __new__(cls, connection_string:Optional[str] = None):
        # If an instance already exists, return it
        if cls._instance is not None:
            # logger.debug(f'SQLManager already exists, returning existing instance to caller {inspect.stack()[1].function}')
            return cls._instance
        
        # If no instance exists, create a new one and store it in _instance
        cls._instance = super().__new__(cls)
        logger.debug('SQLManager does not exist, creating new instance')
        return cls._instance
    def __init__(self, connection_string:Optional[str] = None):
        if hasattr(self, '_engine') and self._engine:
            return

        if connection_string:
            self._connection_string = connection_string
        else:
            # password = os.environ.get('SA_PASSWORD')
            # username = 'sa'
            # server = os.environ.get('Database_Server', 'localhost')
            # database = 'Home_Automation'

            password:str = os.environ.get('SA_PASSWORD') # type: ignore
            assert isinstance(password, str), f'password is not a string, password is: {type(password)}'
            assert len(password) > 0, 'password is empty'

            driver= 'ODBC Driver 17 for SQL Server'
            username = 'sa'

            server = os.environ.get('Database_Server', 'localhost')
            database = 'Home_Automation'

            port = 1433

            self._connection_string = f'DRIVER={driver};SERVER={server};PORT=1433;DATABASE={database};UID={username};PWD={password}'
            self._connection_string = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
            self._connection_string = f"mssql+pyodbc://{username}:{password}@{server}:{port}/{database}?driver={driver}"
            self._connection_string = f"mssql+pyodbc://{username}:{quote_plus(password)}@{server}:{port}/{database}?driver={driver}"


            logger.error(f'driver: {driver} server: {server}, port: {port}, database: {database}, username: {username}, password len: {len(password)}')
        logger.info(f'connecting to SQL Server...')
        try:
            self._engine = create_engine(self._connection_string, echo=False)
        except pyodbc.OperationalError as e:
            logger.critical(f'Failed to connect to SQL Server: {e}')
        # Creating a session factory
        self._session_factory = sessionmaker(bind=self._engine)

        logger.info(f'connected to SQL Server')


    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        # if self._conn exists, close it
        if hasattr(self, '_conn') and self._conn:
            self._conn.close()
    def _send_query(self, sql_query: str, params: Optional[Dict[str, Any]] = None):
        with self._lock:
            with self._engine.connect() as connection:
                if params is not None:
                    # If params are provided, use them in the query
                    try:
                        result = connection.execute(sqlalchemy.text(sql_query), params)
                    except TypeError as e:
                        logger.error(f'Failed to execute query: {sql_query} with params: {params} due to a type error {e}')
                        return None
                    except Exception as e:
                        logger.exception(f'Failed to execute query: {sql_query} with params: {params} due to an error {e}')
                        raise e
                else:
                    # If no params are provided, just execute the query
                    result = connection.execute(sqlalchemy.text(sql_query))
                    
                if result is not None and result.returns_rows:
                    return result.fetchall()
                return None
                
    # def AddFrameDetails(self, frame:Frame, frame_counter_int:int, video_file_path:Path):
    def AddFrameDetails(self, frame_counter_int: int, frame_guid:UUID, video_file_path: Path):
        # Create a new Frame object
        new_frame = SQL_Frame(
            frame_counter_int=frame_counter_int,
            frame_guid=frame_guid,
            video_file_name=str(video_file_path.resolve())
        )

        # Add the new frame to the database
        with self._session_factory() as session:
            try:
                session.add(new_frame)
                session.commit()
            except Exception as e:
                session.rollback()
                logger.error(f'Failed to add frame details to SQL: {e}')
    def DoesVideoExists(self, video_path: Path) -> bool:
        # Ensure we are working with an absolute path
        absolute_video_path = str(video_path.resolve())
        
        with self._session_factory() as session:
            # The session provides a query method directly, no need to use a connection for querying
            try:
                video = session.query(SQL_Video).filter_by(video_file_name=absolute_video_path).first()
            except Exception as e:
                logger.error(f'Failed to query SQL: {e}')
                return False
            return video is not None

    
    def AddVideo(self, video_file_path: Path):
        created_at = datetime.datetime.utcnow()

        new_video = SQL_Video(
            video_file_name=str(video_file_path.resolve()), 
            created_at=created_at
            # Not setting retain_until and deleted_at will leave them as Null
        )

        with self._session_factory() as session:
            try:
                session.add(new_video)
                session.commit()
            except Exception as e:
                session.rollback()
                logger.error(f'Failed to add video to SQL: {e}')

    def DeleteVideo(self, video_file_path: Path):
        deleted_at = datetime.datetime.utcnow()

        sql_command = """
            UPDATE cameras.videos
            SET deleted_at = :deleted_at
            WHERE video_file_name = :video_file_name
        """

        params = {
            'deleted_at': deleted_at,
            'video_file_name': str(video_file_path.resolve())
        }

        try:
            self._send_query(sql_command, params)
        except Exception as e:
            logger.error(f'Failed to delete video from SQL: {e}')

                
    def AddMotion(self, guid:UUID, motion_amount:int):
        return
        # sql definition
        # -- table that stores a log of motion detected. references camera and frame in mssql
        # CREATE TABLE cameras.motion_detected (
        #     motion_detected_id INT NOT NULL IDENTITY(1,1),
        #     motion_amount INT NOT NULL,
        #     frame_guid UNIQUEIDENTIFIER NOT NULL,
        #     created_at DATETIME NOT NULL DEFAULT (GETDATE()),
        #     PRIMARY KEY (motion_detected_id),
        #     FOREIGN KEY (frame_guid) REFERENCES [cameras].[frames](frame_guid)
        # );
        
        # Prepare the SQL query
        sql_command = """
            INSERT INTO cameras.motion_queue (frame_guid, motion_amount)
            VALUES (?, ?)
            """
        self._send_query(sql_command, (str(guid), motion_amount))
    def add_frame_batch(self, batch_data: pd.DataFrame, video_file_path: Path):
        # Writing data to SQL
        try:
            with self._engine.connect() as connection:
                batch_data.to_sql(
                    'frames',
                    con=connection,
                    schema='cameras',
                    index_label='frame_counter_int',
                    if_exists='append',
                    method='multi',
                    chunksize=500  # Adjust as needed
                )
        except Exception as e:
            logger.critical(f'Failed to add frame batch to SQL: {e}')
            #TODO: more specific exception handling