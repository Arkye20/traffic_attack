import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from webapp.utils.utils import get_project_root
import mair



os.chdir(get_project_root())
PATH = "models"

print(mair.__version__)