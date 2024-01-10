import time
from vutils import io


class MyTimer:
    """
    计时器
    """
    def __init__(self, log_file_path: str, init_work: str = "default"):
        self.now_work = init_work
        self.statistic_result = {
            init_work: {
                "total_runtime": 0,
                "max_runtime": 0,
                "min_runtime": float('inf'),
                "iter": 0,
            }
        }
        self.log_file_path = log_file_path

    def __enter__(self):
        print("开始计时...")
        self.start_time = time.time_ns()
        # self.count = 1
        return self

    def __exit__(self, type, value, trace):
        last_time = time.time_ns() - self.start_time
        if last_time > self.statistic_result[self.now_work]['max_runtime']:
            self.statistic_result[self.now_work]['max_runtime'] = last_time
        if last_time < self.statistic_result[self.now_work]['min_runtime']:
            self.statistic_result[self.now_work]['min_runtime'] = last_time
        self.statistic_result[self.now_work]['total_runtime'] += last_time
        self.statistic_result[self.now_work]['iter'] += 1
        for k, v in self.statistic_result.items():
            self.statistic_result[k]['average_runtime'] = v['total_runtime']/v['iter']
        print(f"日志文件保存在{self.log_file_path}")
        io.jsondump(self.statistic_result, self.log_file_path)
        return False

    def label(self, work: str):
        """
        更换工作标签
        """
        last_time = time.time_ns() - self.start_time
        if last_time > self.statistic_result[self.now_work]['max_runtime']:
            self.statistic_result[self.now_work]['max_runtime'] = last_time
        if last_time < self.statistic_result[self.now_work]['min_runtime']:
            self.statistic_result[self.now_work]['min_runtime'] = last_time
        self.statistic_result[self.now_work]['total_runtime'] += last_time
        self.statistic_result[self.now_work]['iter'] += 1
        if work not in self.statistic_result:
            self.statistic_result[work] = {
                "total_runtime": 0,
                "max_runtime": 0,
                "min_runtime": float('inf'),
                "iter": 0,
            }
        self.now_work = work
        self.start_time = time.time_ns()


def get_timer(log_file_path: str, init_work: str = "default") -> MyTimer:
    return MyTimer(log_file_path=log_file_path, init_work=init_work)
