import numpy as np
import scipy.sparse as ss
import  re
import os
from datetime import datetime
from functools import wraps

def take_sparse(k, c):
    chooser = (np.isin(k.nonzero()[0], np.array(c)) * 1) * (np.isin(k.nonzero()[1], np.array(c)) * 1)

    row_indexer = ss.csr_matrix(chooser * k.nonzero()[0]).data
    col_indexer = ss.csr_matrix(chooser * k.nonzero()[1]).data
    k_data = np.ndarray.tolist(k[row_indexer, col_indexer])
    t = dict(zip(c, range(len(c))))

    row_placer = list(map(t.get, row_indexer))
    col_placer = list(map(t.get, col_indexer))

    return ss.csr_matrix((k_data[0], (row_placer, col_placer)))

def extract_numbers(s):
    # Используем регулярное выражение для поиска чисел с учетом знака, включая числа в научной нотации
    numbers = re.findall(r'[+-]?\b\d+\.?\d*(?:[eE][+-]?\d+)?\b', s)
    # Преобразуем найденные строки в числа
    return [float(num) for num in numbers]

class Logger:
    """
    Class for logging
    """
    def __init__(self,active,log_folder,**kwargs):
        self.active=active
        if active:
            self.log_folder=log_folder
            self._init_logging()  # Создаём папку и логируем начальные параметры
            self._log_initial_params(**kwargs)
            self.log_objects=[]
            self.iteration=1
            self.log_string=[]

    def _init_logging(self):
        # Создаём папку ./logs/YYYY-MM-DD_HH-MM-SS/
        self.log_dir = os.path.join(
            self.log_folder,
            datetime.now().strftime("opt_res_%Y-%m-%d_%H-%M-%S")
        )
        os.makedirs(self.log_dir, exist_ok=True)

    def _log_initial_params(self,**kwargs):
        # Сохраняем начальные параметры в initial_params.txt
        with open(os.path.join(self.log_dir, "initial_params.txt"), "w") as f:
            for key in kwargs:
                if kwargs[key] is None: continue
                if isinstance(kwargs[key],np.ndarray):np.save(os.path.join(self.log_dir,f'{key}.npy'),kwargs[key])
                f.write(f'{key}: {kwargs[key]}\n')

    def _set_log_object(self,object: any,write: bool,freq: int,name: str)->None:
        self.log_objects.append((object,write,freq,name))

    @staticmethod
    def log_sub(name: str):
        def _log_sub_(func,):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                #Вызываем оригинальную функцию и получаем результат
                result = func(self, *args, **kwargs)
                #Логируем результат
                if self.active:
                    if isinstance(result,float|int) : self.log_string.append(f'{name} = {result:.3f}, ')
                    if isinstance(result,np.ndarray) : self.log_string.append(f'{name} = {np.array2string(result, precision=3, separator=', ')}, ')

                return result
            return wrapper
        return _log_sub_

    @staticmethod
    def log_end(name: str):
        def _log_end_(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                # Вызываем оригинальную функцию и получаем результат
                result = func(self, *args, **kwargs)

                if self.active:
                    #Запись логов работы функций в файл
                    with open(os.path.join(self.log_dir, "log.txt"), "a") as f:
                        f.write(f'it: {self.iteration}, '+''.join(self.log_string)+f'{name} = {result:.3f}, '+'\n')
                    print(f'it: {self.iteration}, ' + ''.join(self.log_string) + f'{name} = {result:.3f}, ' + '\n')
                    self.log_string=[]

                    #Запись логов переменных задачи
                    for var in self.log_objects:
                        if not (self.iteration%var[2]==0 or self.iteration==1 or self.iteration==16):continue
                        if var[1]:
                            with open(os.path.join(self.log_dir, f"{var[0]}.txt"), "a") as f:
                                f.write(f'it: {self.iteration}, {var[3]} = {getattr(self, var[0], None)} \n')
                        else:
                            np.save(os.path.join(self.log_dir, f"{var[3]}_{self.iteration}.npy"), getattr(self, var[0], None))

                    self.iteration+=1

                return result
            return wrapper
        return _log_end_






