# # import necessary packages and modules
# import datetime
# from unittest import mock
# import numpy as np
# import pytest
# from camera.ImageRecognitionSender import ImageRecognitionSender
# import io

# # import responses module for mocking HTTP requests
# import responses

# # define a fixture for creating an instance of ImageRecognitionSender
# @pytest.fixture
# def sender():
#     # set the IP address to localhost:5000
#     _address = 'http://localhost:5000/'
    
#     # patch the _send_post_request method of the ImageRecognitionSender class with a mock function
#     with mock.patch.object(ImageRecognitionSender, '_send_post_request', wraps=ImageRecognitionSender._send_post_request) as mock_send_post_request:
#         # yield an instance of the ImageRecognitionSender class with the mocked _send_post_request method
#         yield ImageRecognitionSender(ip_address=_address)

# # define a test for the _get_filename method of ImageRecognitionSender
# def test_get_filename(sender):
#     # create a datetime object with a fixed timestamp
#     timestamp = datetime.datetime(2022, 3, 9, 13, 23, 46)

#     # call the _get_filename() method with the fixed timestamp
#     filename = sender._get_filename(timestamp)

#     # assert that the filename is "1646861026.npy"
#     assert filename == "1646861026.npy"

# # define a test for the _priority method of ImageRecognitionSender
# def test_priority(sender):
#     # call the _priority method of the ImageRecognitionSender class and assert that the return value is a string
#     priority = sender._priority()
#     assert isinstance(priority, str)

# # define a test for the _send_post_request method of ImageRecognitionSender
# @responses.activate
# def test_send_post_request(sender):
#     # configure the responses module to return a successful response for POST requests to localhost:5000
#     responses.add(responses.POST, 'http://localhost:5000/', status=200)

#     # call the _send_post_request method of the ImageRecognitionSender class with mock data and files
#     data = {'foo': 'bar'}
#     files = {'file': 'file_data'}
#     response = sender._send_post_request(sender, data=data, files=files)

#     # assert that the response object has a status code of 200
#     assert response.status_code == 200
