import pandas as pd
from typing import List, Union, Any
from ..log import logger


class ExcelHandler:
    def __init__(self, excel_file: str = None, columns: List[str] = None, sheet_name: str = "Sheet1"):
        if excel_file is not None:
            if columns is not None:
                logger.warning("Argument `excel_file` is not None so `columns` will be ignored!")
            self.df = pd.read_excel(excel_file, sheet_name=sheet_name)
        elif columns is not None:
            self.df = pd.DataFrame(columns=columns)
        else:
            raise ValueError("Argument `excel_file` or `columns` must not be None together!")

    def getColumns(self) -> List[str]:
        return self.df.columns.values.tolist()

    def getRowSize(self) -> int:
        return len(self.df)

    def setValue(self, row: int, column: str, value: Union[str, int, float]):
        self.df.loc[row, column] = value

    def getValue(self, row: int, column: str) -> Any:
        return self.df.loc[row, column]

    def fillNanFromMergeUnit(self, column: Union[str, List[str]]):
        if isinstance(column, str):
            column = [column]
        for col in column:
            lastValue = None
            for row in self.df.index:
                if pd.isna(self.df.loc[row, col]) and lastValue is not None:
                    self.df.loc[row, col] = lastValue
                elif not pd.isna(self.df.loc[row, col]):
                    lastValue = self.df.loc[row, col]
                else:
                    raise RuntimeError("Unexpected Nan before all!")

    def saveAs(self, filename: str):
        self.df.to_excel(filename, index=False)

    def __str__(self):
        return str(self.df)
