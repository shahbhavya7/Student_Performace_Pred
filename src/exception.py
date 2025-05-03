import sys
from src.logger import logging

def error_message_detail(error,error_detail:sys): # this function takes error and error_detail as input , error_detail is a module in sys which contains information about the error
    _,_,exc_tb = error_detail.exc_info() # error detail gives 3 values, we are interested in the last one which is exc_tb which contains the traceback of the error
    file_name=exc_tb.tb_frame.f_code.co_filename # extracts the file name from the traceback
    error_message="Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
    file_name,exc_tb.tb_lineno,str(error)) # format fills the values in the string, file_name is the name of the file where the error occurred, 
    # tb_lineno is the line number where the error occurred, str(error) is the error message
    
    return error_message # returns the our own error message with the file name, line number and error message


class CustomException(Exception): # this class inherits from the Exception class, it is a custom exception class
    def __init__(self,error_message,error_detail:sys): # this function takes error_message and error_detail as input, error_detail is a module in sys which contains information about the error
        super().__init__(error_message) # calls the constructor of the parent class Exception with the error_message
        self.error_message=error_message_detail(error_message,error_detail=error_detail) # assign our own error message to the error_message attribute of the 
        # class, this will be used to print the error message when we call object of this class , Exception class originally prints its own error message
        # so we reassign it to our own error message 
    
    def __str__(self):  # __str__ is a special method in python which is called when we print the object of the class, it returns the string representation of the object
        # in this case it returns the error message
        return self.error_message
