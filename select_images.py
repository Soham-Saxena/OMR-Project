from glob import glob
from pathlib import Path
import random
import shutil

categories = ["PNG_GT_BW", "PNG_GT_Gray"]

src = Path("./CVCMUSCIMA_WI/CVCMUSCIMA_WI")
dst = Path("./Separate_Set")

random.seed(42)

for category in categories:
    files = glob(f"./CVCMUSCIMA_WI/CVCMUSCIMA_WI/{category}/**/*00[0-9].png")

    selected = random.sample(files, min(10, len(files)))
    for file in selected:
        name = file.split("\\")[-1]
        shutil.copy2(file, dst/(category + "_" + name))
        