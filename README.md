# vutils

## 安装方法
```shell
pip install vutils
```

## 使用方法

### 文件读写
包括json，jsonl，csv，txt
```python
from vutils import io

data = io.jsonload("data.json")
# do something
io.jsondump(data, "result.json")
```

### Excel文件读写
```python
from vutils.io.excel_handler import ExcelHandler

excel = ExcelHandler("./test.xlsx")
question = excel.getValue(1, '问题')
answer = "yes"
excel.setValue(1, '答案', answer)
excel.saveAs("./result.xlsx")
```

### 日志
```python
from vutils.log import logger

logger.info("你好")
logger.warning("你好")
logger.error("你好")
```

### 网络代理
```python
from vutils.net import set_proxy, unset_proxy

set_proxy(port=7890)
# do something
unset_proxy()
```

### 计时器
```python
import time
import random
from vutils.timer import get_timer

with get_timer(log_file_path="./log.json") as t:
    for i in range(10):
        t.label("生成随机数")
        ts = random.random()
        t.label("打印随机数")
        print(ts)
        t.label("等待时间")
        time.sleep(ts)
```
