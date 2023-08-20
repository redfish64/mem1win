from zipfile import ZipFile 
import re

def dirty_convert_utf8(data):
    return data.decode('utf-8','replace')

# loads txt from gutenberg.org zip file into memory. Will only load if the zip file
# contains a single file with a '.txt' extension and meets the requirements specified by
# the args
def get_etext(zip_filename,min_size=0,max_size=None,start_cutoff_regex=None,end_cutoff_regex=None):
    with ZipFile(zip_filename) as myzip:
        il = myzip.infolist()
        if(len(il) != 1):
            return None,None,f'Expected exactly 1 file in zip, got {len(il)}'
        filename = il[0].filename
        if(not re.search(".txt$",filename)):
            return None,None,f'Filename doesn\'t end with .txt, got {filename}'

        if(il[0].file_size < min_size):
            return None,None,f'Filename less than min_size, filesize {il[0].file_size}'

        if(max_size is not None and il[0].file_size >= max_size):
            return None,None,f'Filename greater or equal to max_size, filesize {il[0].file_size}'

        with myzip.open(filename) as myfile:
            data = myfile.read()

        st_pos = 0
        end_pos = len(data)

        if(start_cutoff_regex is not None):
            m = re.search(data, start_cutoff_regex)
            if(m is not None):
                st_pos = m.end()
            return None,None,f'start cutoff regex not found, {start_cutoff_regex=}'
                
        if(end_cutoff_regex is not None):
            m = re.search(data[st_pos:], end_cutoff_regex)
            if(m is not None):
                end_pos = m.start()
            return None,None,f'end cutoff regex not found, {end_cutoff_regex=}'

        return dirty_convert_utf8(data[st_pos:end_pos]),filename,None

