import xlsxwriter
import random
import glob

def write_excel(formatted_list):
    # Create a workbook and add a worksheet.
    workbook = xlsxwriter.Workbook('spam.xlsx')
    worksheet = workbook.add_worksheet()
    # Start from the first cell. Rows and columns are zero indexed.
    row = 1
    col = 0

    # Iterate over the data and write it out row by row.
    for label, msg in (formatted_list):
        worksheet.write(row, col,     label)
        worksheet.write(row, col + 1, msg)
        row += 1
    workbook.close()

def read_first_line(file, folder_path):
    with open(file, 'rt') as fd:
        first_line = fd.read()
    return [folder_path, first_line]

def merge_per_folder():
    # make sure there's a slash to the folder path 
    folder_path = "ham"
    folder_path1 = "spam"
    formatted_list = []
    # get all text files
    txt_files1 = glob.glob(folder_path + "/*.txt")
    # get all spam text files
    txt_files2 = glob.glob(folder_path1 + "/*.txt")
    # final text file list 
    #txt_files = txt_files1 + txt_files2
    #print(txt_files)
    # get first lines; map to each text file (sorted)
    for file in txt_files1:
        formatted_list.append(read_first_line(file, folder_path))
    for file1 in txt_files2:
        formatted_list.append(read_first_line(file1, folder_path1))
    #print(formatted_list)
    random.shuffle(formatted_list)
    #print(*formatted_list)
    # writing to excel file
    write_excel(formatted_list)
    

if __name__ == "__main__":
    merge_per_folder()