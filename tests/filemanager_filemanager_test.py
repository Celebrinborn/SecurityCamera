import pytest
from camera.filemanager import FileManager, File_Pair
from pathlib import Path
from typing import Union
import os

@pytest.fixture
def file_manager(tmp_path: Path):
    root_folder = tmp_path
    return FileManager(root_folder, max_dir_size=2*24-1)  # space for just less than two file pairs

@pytest.fixture
def file_pair_with_base_filename_path(tmp_path: Path):
    base_filename = tmp_path / "testfile"
    video_file = tmp_path / f"{base_filename}.mp4"
    text_file = tmp_path / f"{base_filename}.txt"
    content = b"some content"  # This will create a file of 12 bytes

    # Write the content to both the files
    with open(video_file, 'wb') as f:
        f.write(content)
    with open(text_file, 'wb') as f:
        f.write(content)
    try:
        yield base_filename, video_file, text_file
    finally:
        # Remove files after test run
        video_file.unlink(missing_ok=True)
        text_file.unlink(missing_ok=True)

@pytest.fixture
def file_pair2(tmp_path: Path):
    base_filename = tmp_path / "testfile2"
    video_file = tmp_path / f"{base_filename}.mp4"
    text_file = tmp_path / f"{base_filename}.txt"
    content = b"some content"  # This will create a file of 12 bytes

    # Write the content to both the files
    with open(video_file, 'wb') as f:
        f.write(content)
    with open(text_file, 'wb') as f:
        f.write(content)

    try:
        yield base_filename, video_file, text_file
    finally:
        # Remove files after test run
        video_file.unlink(missing_ok=True)
        text_file.unlink(missing_ok=True)

def test_scan(file_manager:FileManager, file_pair_with_base_filename_path, file_pair2):
    # Given
    # ... Assume there are some files in file_manager's root folder
    base_filename1:Path
    base_filename2:Path
    base_filename1, video_file1, text_file1 = file_pair_with_base_filename_path
    base_filename2, video_file2, text_file2 = file_pair2

    # When
    file_manager.scan()

    # Then
    # ... Validate the effect of scan()

    assert base_filename1.stem in file_manager.list_pairs().keys(), 'base_filename1 not in filemanager list_pairs keys'
    assert base_filename2.stem in file_manager.list_pairs().keys(), 'base_filename2 not in filemanager list_pairs keys'

     # Given
    base_filename:Path
    base_filename, video_file, text_file = file_pair_with_base_filename_path

    # When
    file_manager.add_file_pair(base_filename)

    # Then
    assert base_filename.stem in file_manager.list_pairs()

def test_delete_file(file_manager:FileManager, file_pair_with_base_filename_path):
    # Given
    base_filename:Path
    video_file:Path
    text_file:Path
    base_filename, video_file, text_file = file_pair_with_base_filename_path
    file_manager.add_file_pair(base_filename)
    # When
    assert video_file.exists(), 'video_file does not exist at start'
    assert text_file.exists(), 'text file does not exist at start'
    assert base_filename.stem in file_manager.list_pairs().keys(), 'base filename not in keys'
    # Then

    file_manager.delete_pair(base_filename)

    assert not video_file.exists(), 'video_file still exists after delete'
    assert not text_file.exists(), 'text file still exists after delete'
    assert base_filename.stem not in file_manager.list_pairs().keys(), 'base filename is still in keys and should no longer be'


def test_add_file(file_manager:FileManager, file_pair_with_base_filename_path):
    # Given
    base_filename:Path
    video_file:Path
    text_file:Path
    base_filename, video_file, text_file = file_pair_with_base_filename_path

    assert len(file_manager.list_pairs()) == 0, 'filemanager is not empty'

    # When
    file_manager.add_file_pair(base_filename)
    
    # Then
    assert len(file_manager.list_pairs()) == 1, 'filemanager did not add file pair'

def test_get_file_size(file_manager:FileManager, file_pair_with_base_filename_path):
    # Given
    base_filename:Path
    video_file:Path
    text_file:Path
    base_filename, video_file, text_file = file_pair_with_base_filename_path

    file_manager.add_file_pair(base_filename)

    # When Then
    assert file_manager.get_pair_file_size(base_filename.stem) == os.path.getsize(video_file) + os.path.getsize(text_file), 'file pair size did not match'

def test_get_total_dir_size(file_manager:FileManager, file_pair_with_base_filename_path, file_pair2):
    # Given
    base_filename:Path
    video_file:Path
    text_file:Path
    base_filename, video_file, text_file = file_pair_with_base_filename_path

    base_filename2:Path
    video_file2:Path
    text_file2:Path
    base_filename2, video_file2, text_file2 = file_pair2

    file_manager.add_file_pair(base_filename)
    file_manager.add_file_pair(base_filename2)

    assert len(file_manager.list_pairs()) == 2
    # When Then
    expected_size = os.path.getsize(video_file) + os.path.getsize(text_file) + os.path.getsize(video_file2) + os.path.getsize(text_file2)
    actual_size = file_manager.get_total_dir_size()
    assert actual_size == expected_size, 'file pair size did not match'
