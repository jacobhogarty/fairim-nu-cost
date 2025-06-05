import pickle
from glob import glob

import asyncio
import aiofiles

from concurrent.futures import ThreadPoolExecutor


class Loader:
    def __init__(self, max_workers=None):
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
        )

    async def load(self, file_path: str):
        """
        Loads a pickle file

        Args:
            file_path: String representing the path of the file

        Returns:
            Loaded pickle file
        """
        async with aiofiles.open(file_path, 'rb') as file:
            data = await file.read()

        return await asyncio.get_event_loop().run_in_executor(
            self.executor,
            pickle.loads,
            data,
        )

    async def load_multiple(self, file_paths: glob, filetype: str):
        """
        Load multiple pickle files concurrently

        Args:
            file_paths: Multiple path object representing the path of the file
            filetype: File extension

        Returns:
            Loaded pickle file
        """
        tasks = [self.load(path) for path in glob(f'{file_paths}/*.{filetype}')]
        return await asyncio.gather(*tasks)

    def __del__(self):
        """
        Destroys thread pool executor after processes complete
        """
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
