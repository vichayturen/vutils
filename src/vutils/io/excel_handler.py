import pandas as pd
from typing import List, Union, Any
from ..log import logger
from ..io import get_extension


class ExcelHandler:
    """
    ExcelHandler is a class to handle excel file
    """
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
        """
        Get columns of excel file
        :return: List of columns
        """
        return self.df.columns.values.tolist()

    def getRowSize(self) -> int:
        """
        Get row size of excel file
        :return: Row size
        """
        return len(self.df)

    def setValue(self, row: int, column: str, value: Union[str, int, float]):
        """
        Set value of excel file
        :param row: Row index
        :param column: Column name
        :param value: Value to set
        """
        self.df.loc[row, column] = value

    def getValue(self, row: int, column: str) -> Any:
        return self.df.loc[row, column]

    def fillNanFromMergeUnit(self, column: Union[str, List[str]]):
        """
        Fill NaN value from last value
        :param column: Column name
        """
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
        """
        Save excel file
        :param filename: File name
        """
        if get_extension(filename) == "xlsx":
            self.df.to_excel(filename, index=False)
        elif get_extension(filename) == "csv":
            self.df.to_csv(filename, index=False)
        else:
            raise ValueError("Unsupported file type!")

    def __str__(self):
        return str(self.df)
