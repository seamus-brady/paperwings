#  Copyright (c) 2024. Prediction By Invention https://predictionbyinvention.com/
#
#  THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
#  INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
#  PARTICULAR PURPOSE, AND NON-INFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
#  COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER
#  IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR
#  IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import logging
import os
import time
import traceback
from typing import Any

from src.paperwings.util.file_path_util import FilePathUtil

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s process:%(process)d %(levelname)s %(message)s",
)


class LoggingUtil:
    """
    Log to a local file, STDOUT and the python logger.
    """

    LOG_DIR = "logs"
    APP_LOG_SUFFIX = "_app.log"

    def __init__(self, prefix: str = "") -> None:
        self.log_file_name: str = self.get_dated_log_file_name()
        self.log_file_dir: str = self.get_log_file_dir()
        self.check_and_create_log_dir()
        if prefix is None:
            self.prefix: str = ""
        else:
            self.prefix = prefix

    def get_dated_log_file_name(self) -> str:
        return time.strftime("%Y%m%d-%H%M%S") + self.APP_LOG_SUFFIX

    def check_and_create_log_dir(self) -> None:
        try:
            if not os.path.exists(self.log_file_dir):
                os.makedirs(self.log_file_dir)
        except:  # noqa #nosec
            pass

    def log(self, log_message: str) -> None:
        # write to local log file
        t: Any = time.localtime()
        log: Any = open(self.get_log_file_path(), "a")
        log_string: str = time.asctime(t) + f"{self.prefix}: " + log_message
        log.write(log_string + "\n")
        log.close()

    def get_log_file_path(self) -> str:
        return os.path.join(self.log_file_dir, self.log_file_name)

    def get_log_file_dir(self) -> str:
        return FilePathUtil.append_path_to_repo_path(self.LOG_DIR)

    def info(self, log_message: str) -> None:
        print(self.prefix + " " + log_message)
        logging.info(self.prefix + " " + log_message)
        self.log("INFO " + log_message)

    def debug(self, log_message: str) -> None:
        print(self.prefix + " " + log_message)
        logging.debug(self.prefix + " " + log_message)
        self.log("DEBUG " + log_message)

    def error(self, log_message: str) -> None:
        backtrace = traceback.format_exc()
        logging.error(self.prefix + " " + log_message)
        logging.error(backtrace)
        print(backtrace)
        print(self.prefix + " ERROR " + log_message)
        self.log("ERROR " + log_message)
        self.log(backtrace)

    @staticmethod
    def instance(prefix="") -> Any:
        return LoggingUtil(prefix=prefix)
