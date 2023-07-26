import sqlite3
import os

import logging
logger = logging.getLogger(__name__)
#     #     sql = '''INSERT INTO detection_results(image_id, category_id, category_name, bbox, area, confidence)
#     #             VALUES(?,?,?,?,?,?) '''
#     #     cur = self.conn.cursor()

#     #     # Iterate over each annotation and insert into the database
#     #     for annotation in results['annotations']:
#     #         try:
#     #             bbox = ",".join(str(b.item()) for b in annotation['bbox'])  # Convert tensor to string
#     #             data = (annotation['image_id'], annotation['category_id'], annotation['category_name'], bbox, annotation['area'].item(), annotation['confidence'])
#     #             cur.execute(sql, data)
#     #         except sqlite3.Error as e:
#     #             logger.error(f"Failed to insert data into the database: {e}.")
#     #             logger.error(f'data was {str(annotation)}')
#     #             continue  # Skip this annotation and try the next one

#     #     try:
#     #         self.conn.commit()
#     #     except sqlite3.Error as e:
#     #         logger.error(f"Failed to commit changes to the database: {e}")
#     #     logger.info("Data inserted successfully into the database")


class HelloWorldContextManager:
    def __enter__(self):
        print("Entering the context")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Exiting the context")

    def helloworld(self):
        print("Hello, World!")