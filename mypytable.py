import copy
import csv
from tabulate import tabulate
import myutils


class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names (list of str): M column names
        data (list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Parameters:
            column_names (list of str): initial M column names (None if empty)
            data (list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure."""
        print(tabulate(self.data, headers=self.column_names))


    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            tuple: (N, M) where N is number of rows and M is number of columns
        """
        return len(self.data), len(self.column_names) 

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Parameters:
            col_identifier (str or int): string for a column name or int
                for a column index
            include_missing_values (bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Raises:
            ValueError: if col_identifier is invalid
        """
        if type(col_identifier) == str:
            if col_identifier not in self.column_names:
                raise ValueError("This column is not found")
            index = self.column_names.index(col_identifier)
        elif type(col_identifier) == int:
            if col_identifier < 0 or col_identifier >= len(self.column_names):
                raise ValueError("Column index not valid")
            index = col_identifier
        else:
            raise ValueError("must be a string or an int")
    
        objects = []
        for i in range(len(self.data)):
            if self.data[i][index] == "NA":
                if include_missing_values == True:
                    objects.append(self.data[i][index])
                else:
                    pass
            else:
                objects.append(self.data[i][index])
        

        return objects


    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leaves values as-is that cannot be converted to numeric.
        """
        for value in range(len(self.data)):
            for j in range(len(self.data[0])):
                try:
                    converted = float(self.data[value][j])
                    self.data[value][j] = converted
                except ValueError as e:
                    pass
            
 
    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Parameters:
            row_indexes_to_drop (list of int): list of row indexes to remove from the table data.
        """
        row_indexes_to_drop.sort(reverse=True)
        for i in row_indexes_to_drop:
            self.data.pop(i)
        # print("New list of data: ", self.data)

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Parameters:
            filename (str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: returns self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Uses the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load.
        """
        table = []
        with open (filename, "r") as infile:
            contents = csv.reader(infile)
            for row in contents:
                table.append(row)
        self.column_names = table[0]
        self.data = table[1:]
        self.convert_to_numeric()
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Parameters:
            filename (str): relative path for the CSV file to save the contents to.

        Notes:
            Uses the csv module.
        """
        with open (filename, "w", newline="") as outfile:
            writing = csv.writer(outfile)
            writing.writerow(self.column_names)
            writing.writerows(self.data)

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Parameters:
            key_column_names (list of str): column names to use as row keys.

        Returns:
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
            The first instance of a row is not considered a duplicate.
        """
        seen = []
        duplicate_indexes = []
        
        for i, row in enumerate(self.data): # to loop through the data (learned enumerate in Ginas)
            key = tuple(row[self.column_names.index(col)] for col in key_column_names)
            if key in seen: 
                duplicate_indexes.append(i)
                print("Duplicate: ", self.data[i])
            else:
                seen.append(key)        

        return duplicate_indexes

    def remove_rows_with_missing_values(self):
        """
        Loops through all rows in the table and deletes any rows with missing values 
        """
        new_table = []
        for row in self.data:
            if "NA" not in row:
                new_table.append(row)
        self.data = new_table
        

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
        by the column's original average.

        Parameters:
            col_name (str): name of column to fill with the original average (of the column).
        """
        
        column_index = self.column_names.index(col_name)
        numeric_values = []
        # had to add because kept having issues!! 
        for row in self.data:
            value = row[column_index]
            if value != "NA":
                try:
                    numeric_values.append(float(value))
                except ValueError:
                    pass  

        # IF NOT EMPTY
        if numeric_values:
            avg = sum(numeric_values) / len(numeric_values)
        else:
            avg = 0  

        # Replace "NA" with average
        for row in self.data:
            if row[column_index] == "NA":
                row[column_index] = avg

   

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Parameters:
            col_names (list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values in the columns to compute summary stats
            should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """

        data = []
        for numbers in col_names:
            column_index = self.column_names.index(numbers)
            column_values = []
            data_index = []

            #what values are we looking at 
            #for i in range(len(self.data)):
                #column_values.append(self.data[i][column_index])
            for row in self.data:
                value = row[column_index]
                if isinstance(value, (int, float)):
                    column_values.append(value)
            
            # adding the min, max, midrange
            if column_values != []:
                column_values = sorted(column_values)
                data_index.append(numbers)
                data_index.append(column_values[0])
                data_index.append(max(column_values))
                data_index.append((min(column_values) + max(column_values))/2)

                # CALCULATES THE AVERAGE
                overall_total = 0
                for value in column_values:
                    if overall_total != "NA":
                        overall_total += value
                avg = overall_total/(len(column_values))
                data_index.append(avg)

                #MEDIAN 
                mid_n = len(column_values) // 2
                if len(column_values) % 2 == 1:
                    median = column_values[mid_n]
                else:
                    median = (column_values[mid_n - 1] + column_values[mid_n]) / 2
                data_index.append(median)


                data.append(data_index)
        new_table = MyPyTable(["attribute", "min", "max", "mid", "avg", "median"], data)
        # print(new_table)
        return new_table 


