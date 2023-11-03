-- table that stores video file names
CREATE TABLE cameras.videos (
    video_file_name unique NVARCHAR(255) NOT NULL,
    created_at DATETIME NOT NULL,
    retain_until DATETIME NOT NULL,
    deleted_at DATETIME NOT NULL,
    PRIMARY KEY (video_file_name)
);

-- table that stores frames. references video_file_name in mssql
CREATE TABLE cameras.frames (
    frame_counter_int INT NOT NULL,
    frame_guid UNIQUEIDENTIFIER unique NOT NULL,
    video_file_name NVARCHAR(255) NOT NULL,
    PRIMARY KEY (frame_guid),
    FOREIGN KEY (video_file_name) REFERENCES cameras.videos(video_file_name)
);

-- table that stores screenshots. references camera_id in mssql
CREATE TABLE cameras.screenshots (
    screenshot_id INT NOT NULL IDENTITY(1,1),
    camera_id INT NOT NULL,
    taken_at DATETIME NOT NULL,
    screenshot_image VARBINARY(MAX) NOT NULL,
    PRIMARY KEY (screenshot_id),
    FOREIGN KEY (camera_id) REFERENCES cameras.cameras(camera_id)
);

-- table that stores camera names
CREATE TABLE cameras.cameras (
    camera_id INT NOT NULL IDENTITY(1,1),
    camera_name NVARCHAR(255) NOT NULL,
    PRIMARY KEY (camera_id)
);

-- table that stores a log of motion detected. references camera and frame in mssql
CREATE TABLE cameras.motion_detected (
    motion_detected_id INT NOT NULL IDENTITY(1,1),
    motion_amount INT NOT NULL,
    frame_guid UNIQUEIDENTIFIER NOT NULL,
    created_at DATETIME NOT NULL DEFAULT (GETDATE()),
    PRIMARY KEY (motion_detected_id),
    FOREIGN KEY (frame_guid) REFERENCES [cameras].[frames](frame_guid)
);


-- motion queue table that stores the frames that have motion detected without it being linked to an existing frame
-- is routinely truncated by the motion queue service as frames are linked to existing frames and moved into the motion_detected table
CREATE TABLE cameras.motion_queue (
    motion_queue_id INT NOT NULL IDENTITY(1,1),
    motion_amount INT NOT NULL,
    frame_guid UNIQUEIDENTIFIER NOT NULL,
    created_at DATETIME NOT NULL DEFAULT (GETDATE()),
    PRIMARY KEY (motion_queue_id),
);
