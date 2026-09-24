import numpy as np
# import matplotlib.pyplot as plt
from scipy import linalg as la
from scipy.sparse.linalg import LinearOperator
# from itertools import permutations, combinations
import scipy.sparse as sp
from scipy.special import comb, factorial
# from scipy import integrate
from pfapack import pfaffian as pf  # computing pfaffian
from scipy import integrate
# import math
import os
import json
# from numba import njit, prange, set_num_threads

class SpinSystem:

    def __init__(self, 
                 width:int=4, height:int=4, 
                 Jz:float=1, Jx:float=1, Jy:float=1,
                 h:float=0, thermo_limit:bool=False,
                 ini_lat_type:str='B'):
        '''
        width and height of the chosen geometric region. The shape of the region is restricted 
        to be rectangular. 
        
        The phyical constituents of the model are spin-1/2 local spins. The lattice is of honeycomb geometry.
        
        ``starting_point`` is the sublattice type of the lattice site sitting at the lower-left corner (labeled as 0)
        '''
        # if width%2 != 0 or height%2 != 0:
        #     raise ValueError('Both width and height are required to be even to ensure a valid PBC')
        
        self._width = int(width)
        self._height = int(height)
        self.N = int(width*height) # total number of sites
        self._dim = int(2**(width*height)) # 2^N, the dimenIon of the total Hilbert space

        self._too_large = False
        if self.N > 7145:
            self._too_large = True

        if not self._too_large:
            self.num_total_config = int(comb(self.N, int(self.N/2)))
        else:
            self.num_total_config = None

        if ini_lat_type in ['A', 'B']:
            self._ini_lat_type = ini_lat_type
        else:
            raise ValueError("``ini_lat_type`` must be 'A' or 'B'")
        
        self._Jx = Jx
        self._Jy = Jy
        self._Jz = Jz
        self._h = h # magnetic field

        self._CoeTable_loaded = False   # if the CoeTable has been loaded, then keep it to avoid repleatedly loeading it again and again.
        self._CoeTable = None

        self._CorrTable_loaded = False   # if the CorrTable has been loaded, then keep it to avoid repleatedly loeading it again and again.
        self._CorrTable = None

        self.full_corr_mat = None

        self._thermo_limit = thermo_limit

        self.approx_bound = None

        self.I = sp.identity(2, format="csr")
        self.X = np.array([[0,1],[1,0]], dtype = np.complex64)
        self.Y = np.array([[0,-1j],[1j,0]], dtype = np.complex64)
        self.Z = np.array([[1,0],[0,-1]], dtype = np.complex64)

        self.rho = np.array([])

        return
    
    @property
    def Jx(self):
        return self._Jx
    @Jx.setter
    def Jx(self, Jx: float):
        self._Jx = Jx
        self.full_corr_mat = None
        self._CoeTable_loaded = False   # reload the CoeTable if the parameter is changed
        self._CoeTable = None
        self._CorrTable_loaded = False
        self._CoeTable = None
        self.approx_bound = None
        return
    
    @property
    def Jy(self):
        return self._Jy
    @Jy.setter
    def Jy(self, Jy: float):
        self._Jy = Jy
        self.full_corr_mat = None
        self._CoeTable_loaded = False   # reload the CoeTable if the parameter is changed
        self._CoeTable = None
        self._CorrTable_loaded = False
        self._CoeTable = None
        self.approx_bound = None
        return
    
    @property
    def Jz(self):
        return self._Jz
    @Jz.setter
    def Jz(self, Jz: float):
        self._Jz = Jz
        self.full_corr_mat = None
        self._CoeTable_loaded = False   # reload the CoeTable if the parameter is changed
        self._CoeTable = None
        self._CorrTable_loaded = False
        self._CoeTable = None
        self.approx_bound = None
        return

    @property
    def h(self):
        return self._h
    @h.setter
    def h(self, h: float):
        self._h = h
        self.full_corr_mat = None
        self._CoeTable = None
        if not os.path.isdir(self.datapath):
            self.calculate_CorrTable()
        self._CoeTable_loaded = False   # reload the CoeTable if the parameter is changed
        self._CorrTable_loaded = False
        self._CoeTable = None
        self.approx_bound = None
        return

    @property
    def thermo_limit(self):
        return self._thermo_limit
    @thermo_limit.setter
    def thermo_limit(self, thermo_limit: bool):
        self._thermo_limit = thermo_limit
        self.full_corr_mat = None
        self._CoeTable_loaded = False
        self._CoeTable = None
        self._CorrTable_loaded = False
        self._CoeTable = None
        self.approx_bound = None
        return

    @property 
    def ini_lat_type(self):
        return self._ini_lat_type
    @ini_lat_type.setter
    def ini_lat_type(self, ini_lat_type:int):
        if not ini_lat_type in ['A', 'B']:
            raise ValueError("``ini_lat_type`` must be 'A' or 'B'")
        self._ini_lat_type = ini_lat_type
        self.full_corr_mat = None
        self._CoeTable_loaded = False   # reload the CoeTable if the parameter is changed
        self._CoeTable = None
        self._CorrTable_loaded = False
        self._CoeTable = None
        self.approx_bound = None
        return
    
    @property
    def datapath(self):
        if self.ini_lat_type == 'B':   # the lower-left corner is type-B
            file_path = "data/w_%d_h_%d/Jx_%.3f_Jy_%.3f_Jz_%.3f_mag_%.3f" % (self.width, self.height, self.Jx, self.Jy, self.Jz, self.h)
        else:
            file_path = "data_init_is_A/w_%d_h_%d/Jx_%.3f_Jy_%.3f_Jz_%.3f_mag_%.3f" % (self.width, self.height, self.Jx, self.Jy, self.Jz, self.h)
        return file_path
    
    @property
    def width(self):
        return self._width
    @width.setter
    def width(self, width:int):
        self._width = width
        self.N = int(self._width * self._height)
        self.full_corr_mat = None
        self._CoeTable_loaded = False
        self._CoeTable = None
        self._CorrTable_loaded = False
        self._CoeTable = None
        return

    @property
    def height(self):
        return self._height
    @height.setter
    def height(self, height:int):
        self._height = height
        self.N = int(self._width * self._height)
        self.full_corr_mat = None
        self._CoeTable_loaded = False
        self._CoeTable = None
        self._CorrTable_loaded = False
        self._CoeTable = None
        return

    @property
    def dim(self):
        self._dim = int(2**(self.width * self.height))
        return self._dim

    @property
    def CoeTable_loaded(self):  # load CoeTable once it is called
        initial = self._CoeTable_loaded # store the initial situation so that it won't be overwritten
        if not self._CoeTable_loaded:   # load CoeTable if it has not been loaded
            with open(self.datapath + '/CoeTable.json', 'r') as file:
                self._CoeTable = json.load(file)
        self._CoeTable_loaded = True
        return initial
    
    ################### correlation functions ###################
    """
    Here, the momentum is expressed in terms of reciprocal basis vectors
    b_1 = (2pi, 2pi/sqrt(3))
    b_2 = (0, 4pi/sqrt(3))
    as
    k = k1 b_1 + k2 b_2
    which are derived by the lattice basis vectors 
    a_1 = (1,0)
    a_2 = (1/2, sqrt(3)/2)
    """
    def sk(self, k1:float ,k2:float)->float:
        Jx = self.Jx
        Jy = self.Jy
        Jz = self.Jz
        return 2*np.sqrt(Jx**2 + Jy**2 + Jz**2 + 
                    2*Jx*Jy*np.cos(2*np.pi*k1) + 
                    2*Jx*Jz*np.cos(2*np.pi*k2) + 
                    2*Jy*Jz*np.cos(2*np.pi*(k1 - k2)))
    
    def Rk(self, k1:float, k2:float)->float:
        Jx = self.Jx
        Jy = self.Jy
        Jz = self.Jz
        return 2*(Jx*np.cos(2*np.pi*k2) + Jy*np.cos(2*np.pi*(k1 - k2)) + Jz)
    
    def Ik(self, k1:float, k2:float)->float:
        Jx = self.Jx
        Jy = self.Jy
        return 2*(Jx*np.sin(2*np.pi*k2) - Jy*np.sin(2*np.pi*(k1 - k2)))

    def Delta(self, k1:float, k2:float)->float:
        Jx = self.Jx
        Jy = self.Jy
        Jz = self.Jz
        J = (Jx + Jy + Jz)/3    # average of the spin coupling strength
        h = self.h  # external magnetic field
        kappa = h**3 / J**2

        return  2*kappa*(np.sin(2*np.pi*k2) - 
                         np.sin(2*np.pi*(k1-k2)) +
                         np.sin(2*np.pi*k1))
    
    def ep(self, k1:float, k2:float, eta:float=1e-3)->float:
        return np.sqrt((self.sk(k1=k1, k2=k2))**2 + (self.Delta(k1=k1, k2=k2))**2 + eta**2)
    
    def corr_integrand(self, k1:int, k2:int, dx:int, dy:int, pair_type:str)->float:
        """
        ``pair_type`` = 'AA', 'BB', 'AB', 'BA'
        """
        R, I, epsilon, Delta = self.Rk(k1=k1, k2=k2), self.Ik(k1=k1, k2=k2), self.ep(k1=k1, k2=k2), self.Delta(k1=k1, k2=k2)
        if pair_type == 'AA':
            phase = 2*np.pi*(k1*dx + k2*dy)
            integrand = - Delta * np.sin(phase) / epsilon
        elif pair_type == 'BB':
            phase = 2*np.pi*(k1*dx + k2*dy)
            integrand = Delta * np.sin(phase) / epsilon
        elif pair_type == 'AB':
            phase = 2*np.pi*(k1*dx + k2*dy)
            integrand = (R*np.cos(phase) + I*np.sin(phase)) / epsilon
        elif pair_type == 'BA':
            phase = - 2*np.pi*(k1*dx + k2*dy)   # the relative position is also flipped after exchanging the position of the two majorana fermion operators
            integrand = - (R*np.cos(phase) + I*np.sin(phase)) / epsilon
        else:
            raise ValueError("``pair_type`` must be 'AA', 'BB', 'AB', or 'BA'.")
        return integrand

    def corr_thermo_limit_engine_c(self, dx:int, dy:int, pair_type:bool)->float:
        """
        Two-points correlation function <-ic_{n1}c_{n2}>. Notice the existance of the imaginary number -i
        """
        int_value, err = integrate.dblquad(self.corr_integrand, 
                                            0, 1, 
                                            0, 1, 
                                            args=(dx, dy, pair_type))
        return  int_value
        
    def corr_thermo_limit_engine(self, n1:int, n2:int)->float:
        """
        Two-points correlation function <-ic_{n1}c_{n2}>. Notice the existance of the imaginary number -i
        """
        x1, y1, mu1 = self.dictionary_n_to_c(label=n1) 
        x2, y2, mu2 = self.dictionary_n_to_c(label=n2)
        dx, dy = x2 - x1, y2 - y1 

        if n1==n2:  # A Majorana fermion operator squares to one. We don't need this value.
            return None
        else:       # The two Majorana fermions are at different positions
            x1, y1, mu1 = self.dictionary_n_to_c(label=n1) 
            x2, y2, mu2 = self.dictionary_n_to_c(label=n2)
            dx, dy = x2 - x1, y2 - y1 
            if mu1 == 0 and mu2 == 0:
                pair_type = 'BB'
            elif mu1 == 0 and mu2 == 1:
                pair_type = 'BA'
            elif mu1 == 1 and mu2 == 0:
                pair_type = 'AB'
            elif mu1 == 1 and mu2 == 1:
                pair_type = 'AA'
            else:
                raise ValueError('Something went wrong.')
            return self.corr_thermo_limit_engine_c(dx=dx, dy=dy, pair_type=pair_type)

    def corr_engine_c(self, dx:int, dy:int, pair_type:bool, tot_size:list[int,int]=None)->float:
        """
        Two-points correlation function <-ic_{n1}c_{n2}>. Notice the existance of the imaginary number -i
        ``tot_size`` = [width, height] is the total system size. Note that the total system size need not to be identical to the system size we set in the beginning since the system size may be merely a subsystem. 
        The default value of ``tot_size`` is [self.width, self.height] we set in the beginning.
        Notice that the value of ``tot_size[1]`` (height) can be 'infty', which means the entire system is a infinitly long cylinder streching along z-direction.
        """
        if tot_size is None:    # default: the total size is exactly the system size we set in the beginning with PBC
            tot_size = [self.width, self.height]
            tot_num = tot_size[0] * tot_size[1]
        else:
            if tot_size[1] == 'infty':  # infinitly long cylinder streching along z-direction.
                tot_num = tot_size[0]
            else:
                tot_num = tot_size[0] * tot_size[1]

        if tot_size[1] == 'infty':
            corr_fun = 0
            for nx in range(int(tot_size[0]/2)):
                k1 = nx / (tot_size[0]/2)
                # R, I, epsilon, Delta = self.Rk(k1=k1, k2=k2), self.Ik(k1=k1, k2=k2), self.ep(k1=k1, k2=k2), self.Delta(k1=k1, k2=k2)
                int_func = lambda k: self.corr_integrand(k1=k1, k2=k, 
                                                            dx=dx, dy=dy,
                                                            pair_type=pair_type)
                val, err = integrate.quad(int_func, 0, 1)
                corr_fun += val
        else:
            corr_fun = 0
            for nx in range(int(tot_size[0]/2)):
                for ny in range(tot_size[1]):
                    k1 = nx / (tot_size[0]/2)
                    k2 = ny / tot_size[1]
                    # R, I, epsilon, Delta = self.Rk(k1=k1, k2=k2), self.Ik(k1=k1, k2=k2), self.ep(k1=k1, k2=k2), self.Delta(k1=k1, k2=k2)
                    corr_fun += self.corr_integrand(k1=k1, k2=k2, 
                                                    dx=dx, dy=dy,
                                                    pair_type=pair_type)
        return (2/tot_num) * corr_fun

    def corr_engine(self, n1:int, n2:int, tot_size:list[int,int]=None)->float:
        """
        Two-points correlation function <-ic_{n1}c_{n2}>. Notice the existance of the imaginary number -i
        ``tot_size`` = [width, height] is the total system size. Note that the total system size need not to be identical to the system size we set in the beginning since the system size may be merely a subsystem. 
        The default value of ``tot_size`` is [self.width, self.height] we set in the beginning.
        """
        x1, y1, mu1 = self.dictionary_n_to_c(label=n1) 
        x2, y2, mu2 = self.dictionary_n_to_c(label=n2)
        dx, dy = x2 - x1, y2 - y1 

        if n1==n2:  # A Majorana fermion operator squares to one. We don't need this value.
            return None
        else:       # The two Majorana fermions are at different positions
            if mu1==1 and mu2==1:
                pair_type = 'AA'
            elif mu1==0 and mu2==0:
                pair_type = 'BB'
            elif mu1==1 and mu2==0:
                pair_type = 'AB'
            else:
                pair_type = 'BA'
            return self.corr_engine_c(dx=dx, dy=dy, pair_type=pair_type, tot_size=tot_size)
    
    def calculate_CorrTable(self, tot_size:list[int,int]=None,
                            corr_table_sys_size:list[int,int]=None):
        """
        A dictionary with two-point correlation functions stored within it
        """
        print('\033[34mCreating CorrTable\033[0m')

        if not corr_table_sys_size is None:
            width_origin, height_origin = self.width, self.height
            self.width, self.height = corr_table_sys_size[0], corr_table_sys_size[1]

        def corr_func(dx, dy, pair_type):
            # if the variable self.thermo_limit is True, then tot_size won't be taken, no matter we set it or not.
            if self.thermo_limit:   
                return self.corr_thermo_limit_engine_c(dx=dx, dy=dy, pair_type=pair_type)
            else:
                return self.corr_engine_c(dx=dx, dy=dy, pair_type=pair_type, tot_size=tot_size)
            
        CorrTable = {}
        for x in np.arange(-int(self.width/2), int(self.width) + 1):
            for y in np.arange(-int(self.height), int(self.height)+1):
                # print('\033[34m%d-th corr function\033[0m' % n)
                # AB pairing
                CorrTable['(%d,%d)AB'%(x,y)] = corr_func(dx=x, dy=y, pair_type='AB') 
                # BA pairing
                CorrTable['(%d,%d)BA'%(x,y)] = corr_func(dx=x, dy=y, pair_type='BA') 
                # AA pairing
                CorrTable['(%d,%d)AA'%(x,y)] = corr_func(dx=x, dy=y, pair_type='AA') 
                # BB pairing
                CorrTable['(%d,%d)BB'%(x,y)] = corr_func(dx=x, dy=y, pair_type='BB') 
        
        if self.thermo_limit:
            filename = 'CorrTable_thermo_limit.json'
        else:
            if tot_size is None:    # default
                filename = 'CorrTable.json'
            else:
                if tot_size[1] == 'infty':
                    filename = f'CorrTable_totsize_{tot_size[0]}_infty.json'
                else:
                    filename = f'CorrTable_totsize_{tot_size[0]}_{tot_size[1]}.json'
        os.makedirs(name=self.datapath, exist_ok=True)
        with open(self.datapath  + '/' + filename, "w") as file:
            json.dump(CorrTable, file)  
        print('\033[34mCorrTable creation complete\033[0m')

        if not corr_table_sys_size is None:
            self.width, self.height = width_origin, height_origin

        return  CorrTable
    
    def corr(self, n1:int, n2:int, update:bool=False, tot_size:list[int,int]=None,
             corr_table_sys_size:list[int,int]=None)->float:
        """
        Two-points correlation function <-icc>. Notice the existance of the imaginary number -i
        This function is valid only for PBC
        """
        if self.thermo_limit:   # if the variable self.thermo_limit is True, then tot_size won't be taken, no matter we set it or not.
            filename = 'CorrTable_thermo_limit.json'
        else:
            if tot_size is None:    # default
                filename = 'CorrTable.json'
            else:
                if tot_size[1] == 'infty':
                    filename = f'CorrTable_totsize_{tot_size[0]}_infty.json'
                else:
                    filename = f'CorrTable_totsize_{tot_size[0]}_{tot_size[1]}.json'

        if not corr_table_sys_size is None: # temporary reset the lattice size
            width_origin, height_origin = self.width, self.height
            self.width, self.height = corr_table_sys_size[0], corr_table_sys_size[1]

        if update or not os.path.isfile(self.datapath+'/'+filename):
            self.calculate_CorrTable(tot_size=tot_size, 
                                     corr_table_sys_size=corr_table_sys_size)
            if self.thermo_limit:
                corr_value = self.corr_thermo_limit_engine(n1=n1, n2=n2)
            else:
                corr_value = self.corr_engine(n1=n1, n2=n2)
        else:
            if self._CorrTable_loaded:          # use the loaded correlation table directly to avoid opening the CorrTable repeatedly
                CorrTable = self._CorrTable
            else:
                with open(self.datapath + '/' + filename, "r") as file:
                    CorrTable = json.load(file)
                self._CorrTable = CorrTable     # update self._CorrTable
                self._CorrTable_loaded = True   # update the status

            x1, y1, mu1 = self.dictionary_n_to_c(label=n1) 
            x2, y2, mu2 = self.dictionary_n_to_c(label=n2)
            dx, dy = x2 - x1, y2 - y1 

            if mu1==1 and mu2==0:   # AB pairing
                corr_value = CorrTable['(%d,%d)AB'%(dx,dy)]
            elif mu1==0 and mu2==1:   # BA pairing
                corr_value = CorrTable['(%d,%d)BA'%(dx,dy)]
            elif mu1==1 and mu2==1:   # AA pairing
                corr_value = CorrTable['(%d,%d)AA'%(dx,dy)]
            else:                      # BB pairing
                corr_value = CorrTable['(%d,%d)BB'%(dx,dy)]

        if not corr_table_sys_size is None: # restore the system size
            self.width, self.height = width_origin, height_origin

        return corr_value
    
    def full_corr_mat_update(self, tot_size:list[int,int]=None, update:bool=False,
                             corr_table_sys_size:list[int,int]=None):
        '''
        C_{ij} =    0           ;   i = j
                =    <-ic_ic_j>  ;   i neq j
        '''
        # self.calculate_CorrTable()
        corr_mat = np.zeros((self.N, self.N))
        tot_amount = (self.N + 1)*self.N / 2
        progress = 1
        count = 0
        print('\033[34mProgress: Making Correlation Matrix\033[0m')
        if update:
            self.calculate_CorrTable(tot_size=tot_size, 
                                     corr_table_sys_size=corr_table_sys_size)
        for i in np.arange(self.N):
            for j in np.arange(i, self.N):
                if i == j:
                    corr_mat[i,i] = 0
                else:
                    # NOTE: if self.thermo_limit is True, the tot_size won't be taken
                    corr_mat[i,j] = self.corr(n1=i, n2=j, 
                                              tot_size=tot_size,
                                              corr_table_sys_size=corr_table_sys_size)
                    corr_mat[j,i] = - corr_mat[i,j]

                # progress report
                count += 1
                if int(10 * count / tot_amount) >= progress:
                # if int(10 * reps_count / self.dim) >= progress:
                    print('\033[34mProgress: %d/10\033[0m' % (progress))
                    progress += 1

        self.full_corr_mat = corr_mat
        return None

    def Coefficient_engine(self, label:int, tot_size:list[int,int]=None,
                           corr_table_sys_size:list[int,int]=None)->list[int, float]:
        '''
        Return: (number of TRodd, pfaffian of the correlation matrix)
        Note that (i)^{number of TRodd} = phase factor
        Calculate the coefficient (the expectation value of string operators) in front of each basis (configuration) by Wick's theorem.
        The coefficient is proportional to the pfaffian of the correlation matrix

        cor_mat_{ij} =  0       ; i = j
                     =  <-icc>  ; i neq j
        '''
        # create the correlation matrix (skew-symmetric), which is a submatrix of the full correlation matrix 
        setA, setB = label_to_endpointset(label=label)
        endpointset = setA + setB
        endpointset.sort()
        endpointset = np.array(endpointset, dtype=np.int64)
        if len(endpointset)%2 == 1:
            raise ValueError('There are supposed to be an even number of endpoints in total.')

        if self.full_corr_mat is None:  # create the correlation matrix if it has not had one
            self.full_corr_mat_update(tot_size=tot_size, corr_table_sys_size=corr_table_sys_size)

        cor_pfa = corr_pfaffian_calculator(mat=self.full_corr_mat, endpointset=endpointset)
        
        num_trodd_strings = int(np.abs(len(setA) - len(setB)) / 2)  # the number of TRodd strings, which are strings with both endpoints in the same sublattice   

        return [num_trodd_strings, cor_pfa]

    def calculate_CoefficientTable(self, update:bool=False, tot_size:list[int,int]=None,
                                   corr_table_sys_size:list[int,int]=None)->dict:
        """
        NOTE: If the region is exactly the entire system with PBC, please do not set the variable ``tot_size``, because we can take advantage of the PBC to reduce computational cost.
        
        {
            config: (num of TRodd, pfaffian of the correlation matrix)
        }
        The label `config` is the decimal representation of the binary representation of the configuration of occupied lattice sites.
        `standard` means outputing the standard table: Jx=Jy=Jz=1 
        """
        path = self.datapath
        # if not standard:
        #     path = self.datapath
        # else:
        #     path = "data/w_%d_h_%d/Jx_%.3f_Jy_%.3f_Jz_%.3f" % (self.width, self.height, 1, 1, 1)

        if self.thermo_limit:
            filename = 'CoeTable_thermo_limit.json'
        else:
            if tot_size is None:
                filename = 'CoeTable.json'
            else:
                if tot_size[1] == 'infty':
                    filename = f'CoeTable_totsize_{tot_size[0]}_infty.json'
                else:
                    filename = f'CoeTable_totsize_{tot_size[0]}_{tot_size[1]}.json'

        if not os.path.isfile(path + '/' + filename) or update:
            print('\033[34mCreating coefficient table for Jx=%.3f, Jy=%.3f, Jz=%.3f, mag=%.3f\033[0m' % (self.Jx, self.Jy, self.Jz, self.h))
            
            CoeTableDict = {}
            progress = 1
            if tot_size is None and self.thermo_limit==False:    # default mode: the region is exactly the entire system, so it has PBC.
                reps = self.representative_set()
                reps_count = 0
                for D in reps:   # we have PBC in both directions
                    setA, setB = label_to_endpointset(label=D)
                    if len(setA + setB)%2 == 1: # the number of endpoints must be even.
                        reps_count += 1
                        continue

                    CoeTableDict[D] = self.Coefficient_engine(label=D, corr_table_sys_size=corr_table_sys_size)

                    # configurations differ by a translation 
                    for move_x in range(int(self.width/2)):
                        for move_y in range(int(self.height)):
                            D_move = self.translation_label(label=D, 
                                                            move=[move_x, move_y])
                            if D_move == D:
                                pass
                            else:
                                CoeTableDict[D_move] = CoeTableDict[D]
                    
                    # progress report
                    reps_count += 1
                    if int(10 * reps_count / len(reps)) >= progress:
                    # if int(10 * reps_count / self.dim) >= progress:
                        print('\033[34mProgress: %d/10\033[0m' % (progress))
                        progress += 1
            else:
                count = 0
                for D in range(self.dim):
                    setA, setB = label_to_endpointset(label=D)
                    if len(setA + setB)%2 == 1: # the number of endpoints must be even.
                        count += 1
                        continue
                    CoeTableDict[D] = self.Coefficient_engine(label=D, corr_table_sys_size=corr_table_sys_size)
                    # progress report
                    count += 1
                    if int(10 * count / self.dim) >= progress:
                        print('\033[34mProgress: %d/10\033[0m' % (progress))
                        progress += 1
                        
            os.makedirs(path, exist_ok=True)
            with open(path + '/' + filename, 'w') as file:
                json.dump(CoeTableDict, file)
        else:
            with open(path + '/' + filename, 'r') as file:
                CoeTableDict = json.load(file)

        return CoeTableDict

    def Coefficient(self, label:int, tot_size:list[int,int]=None)->tuple[complex, float]:
        '''
        Return the coefficient in front of each basis (configuration) by Wick's theorem.
        The output is a tuple (phase factor, pfaffian coefficient)

        <(-ic_{a1} c_{b1}) (-ic_{a2} c_{b2})... > = sum over all possible parings of 
        <-ic_{ai} c_{bj}> products.
        The coefficients can be expressed as the determinents of correlation matrices defined as G_{ij} = <c_Ai c_Bj>
        The coefficient in front of each basis (configuration). 
        siteA and siteB are np.arrays, whose elements are end points of the configuration.

        We should be careful that <(-ic1c2)(-ic3c4)> changes sign when we permute the fermionic operators to <(-ic1c4)(-ic3c2)>.

        Note that the correlators are purely imaginary, so the calculation can be done under dtype=float 
        '''
        if self.thermo_limit:
            filename = 'CoeTable_thermo_limit.json'
        else:
            if tot_size is None:
                filename = 'CoeTable.json'
            else:
                if tot_size[1] == 'infty':
                    filename = f'CoeTable_totsize_{tot_size[0]}_infty.json'
                else:
                    filename = f'CoeTable_totsize_{tot_size[0]}_{tot_size[1]}.json'

        if not os.path.isfile(self.datapath + '/' + filename):    # Create CoeTable first if it doesn't exists in the disk
            self.calculate_CoefficientTable()
            setA, setB = label_to_endpointset(label=label)
            return self.Coefficient_engine(setA=setA, setB=setB) 
        else:                                                   # load CoeTable from the disk if it exists.
            if self.CoeTable_loaded:            # if CoeTable has been loaded, then use the the table kept one instead of loaded it again and again. 
                                                # Note that self._CoeTable_loaded will turn to be True once self.CoeTable_loaded is called.
                coe_dict = self._CoeTable
            else:
                with open(self.datapath + '/CoeTable.json', 'r') as file:
                    coe_dict = json.load(file)
                    self.CoeTable_loaded
            return coe_dict[f'{label}']
        
    ################### dictionary ###################

    def dictionary_n_to_l(self,label:int)->tuple[int, int]:
        '''
        Input the label of the site (an integer) and then ouput the layer-site representation of 
        the site (layer, site).
        '''
        label = int(label)
        site = label%self.width
        layer = int(label/self.width)

        return layer, site
    
    def dictionary_l_to_n(self, layer:int, site:int)->int:
        '''
        Inverse function of dictionary_n_to_l
        '''
        # periodic boundary condition
        layer, site = layer % self.height, site % self.width

        label = self.width*layer + site

        return label
    
    def dictionary_n_to_c(self, label:int)->tuple[int, int, int]:
        '''
        Input the label of the site (an integer) then output the coordinate representation of 
        the site (\vec{x}, \mu). 

        Components of \vec{x} are integers m1 and m2 such that the poItion of the Bravais 
        lattice site is 
        m1\vec{a1} + m2\vec{a2}
        , where 
        \vec{a1} = (1,0)
        \vec{a2} = (1/2, \sqrt{3}/2) 

        \mu = 0 stands for B-sublattice 
            = 1 stands for A-sublattice
        '''
        if self.ini_lat_type == 'B':
            mu = label % 2
        else:
            mu = (label+1) % 2 

        layer, site = self.dictionary_n_to_l(label=label)
        
        # site = 2 * sq + sr
        sr = site % 2
        sq = int(site/2)

        if self.ini_lat_type == 'B':
            m1 = sq + sr    # x = sq + sr
            m2 = layer - sr # y = l - sr
        else:
            m1 = sq
            m2 = layer + sr 

        return m1, m2, mu
    
    def dictionary_l_to_c(self, layer:int, site:int)->tuple[int, int, int]:
        # periodic boundary condition
        layer, site = layer % self.height, site % self.width

        label = self.dictionary_l_to_n(layer=layer, site=site)
        m1, m2, mu = self.dictionary_n_to_c(label=label)

        return m1, m2, mu
    
    def SubLatID_l(self, layer:int, site:int)->int:
        """
        identify the sublattice
        """
        n = self.dictionary_l_to_n(layer=layer, site=site)
        return n%2
    
    ################### foundamental operators ###################
    
    def one_spin(self,length:int, op:np.ndarray, site:int)->np.ndarray:
        """
        Put an operator "op" on "site"
        """
        L = length
        I = self.I

        A = None
        for j in range(L):
            Aj = op if j == site else I
            A = Aj if A is None else sp.kron(A, Aj, format="csr") # A being None means we are at the first site, so we don't have anyone to kron prod yet.
        
        return A

    def two_spin(self, length:int, site1:int, op1:np.ndarray, site2:int, op2:np.ndarray)->np.ndarray:
        """
        Put "op1" and "op2" on "site1" and "site2" respectively
        """
        L = length
        I = self.I

        A = None
        for site in range(L):
            if site == site1:
                Aj = op1
            elif site == site2:
                Aj = op2
            else:
                Aj = I
            
            A = Aj if A is None else sp.kron(A, Aj, format="csr")
        
        return A
    
    def PauliString(self, pauliword:dict)->np.ndarray:
        '''
        Construct a corresponding Pauli string operator according to the given pauliword.

        ### Pauliword
        The format of a pauliword is like

        {
            sitelabel_1: type_1
            sitelabel_2: type_2
            ...
        }
        where each sitelabel_i is an integer and each type_i is 'X', 'Y', or 'Z'
        '''
        matrix = None
        for i in range(self.N):
            if i in pauliword:
                if pauliword[i] == 'X':
                    s = self.X
                elif pauliword[i] == 'Y':
                    s = self.Y
                else:   # pauliword[i] == 'Z'
                    s = self.Z
            else:
                s = self.I
            matrix = s if matrix is None else sp.kron(matrix,s, format='csr')
        
        return matrix
    
    ################### string operator ###################
    
    def dimer_direction(self, n1: int, n2:int)->str:
        """
        Input two nearest neighboring sites n1 and n2, and then return the direction of the dimer, which will be 'X', 'Y' or 'Z'.
        """
        # n1 is type-A by default. Thus, if the situation is opposite, then we have to switch it to our setting
        mu1, mu2 = n1%2, n2%2
        if mu1 == 1 and mu2 == 0:
            pass
        elif mu1 == 0 and mu2 == 1:
            n1_temp = n1
            n1 = n2
            n2 = n1_temp
        else:
            raise ValueError('two endpoints of a valid string must be of different types')
        
        l1, s1 = self.dictionary_n_to_l(label=n1)
        l2, s2 = self.dictionary_n_to_l(label=n2)

        # determine the direction of the dimer
        if l2%self.height == (l1-1)%self.height and s1%self.width == (s2-1)%self.width:
            return 'Z'
        elif s2%self.width == (s1 + 1)%self.width and l1%self.height == l2%self.height:
            return 'X'
        elif s2%self.width == (s1 - 1)%self.width and l1%self.height == l2%self.height:
            return 'Y'
        else:
            raise ValueError("Invalid points for forming a dimer")
    
    def dimer(self, n1: int, n2:int)->np.ndarray:
        """
        Input two nearest neighboring sites n1 and n2, and then return the dimer operator in the Kitaev model \dimer(n1,n2)
        """
        dir = self.dimer_direction(n1=n1, n2=n2)
        
        if dir == 'Z':
            return self.two_spin(length=self.N, site1=n1, site2=n2, op1=self.Z, op2=self.Z)
        elif dir == 'X':
            return self.two_spin(length=self.N, site1=n1, site2=n2, op1=self.X, op2=self.X)
        elif dir == 'Y':
            return self.two_spin(length=self.N, site1=n1, site2=n2, op1=self.Y, op2=self.Y)
        else:
            raise ValueError("Invalid points for forming a dimer")
    
    def string_op_pauliword(self, na:int, nb:int)->tuple[np.complex64, dict]:
        """
        Generate the Pauli *words* of a single string, whose endpoints are na (in sublattice A) and nb (in sublattice B)
        The output is (ImagCoe, pauliword). The phase factor ImagCoe stems from the commutation relations of the Pauli operators.
        
        ### Pauliword
        The format of a pauliword is like

        {
            sitelabel_1: type_1
            sitelabel_2: type_2
            ...
        }
        where each sitelabel_i is an integer and each type_i is 'X', 'Y', or 'Z'
        """
        pauliword = {}

        la, sa = self.dictionary_n_to_l(label=na)
        lb, sb = self.dictionary_n_to_l(label=nb)
        
        l, s = la, sa   # initialize the moving point. RULE: always starts from the type-A endpoint

        ImagCoe_tot = 1 # the imaginary factor due to the Pauli algebra
        
        pauli_type = None   # the type of the Pauli spin we want to add at the current site
        while l != lb or s != sb:

            ########### move the point to generate (l_next, s_next) ###########
            if lb == l%self.height:
                if sb > s%self.width:
                    l_next, s_next = l, s+1
                    #print('right')
                else:   #sb < s
                    l_next, s_next = l, s-1
                    #print('left')
            elif lb > l%self.height:
                if self.SubLatID_l(layer=l, site=s) == 0:   # sublattice B
                    l_next, s_next = l+1, s-1
                    #print('upper left')
                elif self.SubLatID_l(layer=l, site=s) == 1:   # sublattice A
                    l_next, s_next = l, s+1
                    #print('right')
            elif lb < l%self.height:
                if self.SubLatID_l(layer=l, site=s) == 0:   # sublattice B
                    l_next, s_next = l, s-1
                    #print('left')
                elif self.SubLatID_l(layer=l, site=s) == 1:   # sublattice A
                    l_next, s_next = l-1, s+1
                    #print('lower right')
            else:
                raise ValueError('invalid algorithm')
            ####################################################################
            
            # represent the points in terms of labeling numbers
            n, n_next = self.dictionary_l_to_n(layer=l, site=s), self.dictionary_l_to_n(layer=l_next, site=s_next)
            
            # determine the pauli type of the spin operator we want to add,
            if pauli_type is None:  # initial step
                dir = self.dimer_direction(n1=n, n2=n_next) # the direction of the current dimer
                #print('dir = ' + dir)
                pauli_type = dir    # at the first step, the Pauli spin at the initial point is in the same direction as the initial dimer, and it does not intersects with other Pauli spins
            else:
                dir_next = self.dimer_direction(n1=n, n2=n_next)
                #print('dir = ' + dir)
                #print('dir_next = ' + dir_next)
                ImagCoe, pauli_type = PauliMultRule(type1=dir, type2=dir_next) # equivalent to multiplying two dimer operators
                #print('pauli_type = ' + pauli_type)
                ImagCoe_tot = ImagCoe_tot * ImagCoe # update the imaginary coefficient
                dir = dir_next  # update the dimer direction
            
            #  collect the pauli type into the sets
            if pauli_type in ['X', 'Y', 'Z']:
                pauliword[n] = pauli_type
            else:
                raise ValueError('invalid algorithm for constructing string operators')
            
            l, s = l_next%self.height, s_next%self.width   # update

        # final step: for the terminal endpoint
        pauli_type = dir    # at the final step, the Pauli spin at the last point is in the same direction as the last dimer, and it does not intersects with other Pauli spins
        #  collect the pauli type into the sets
        if pauli_type in ['X', 'Y', 'Z']:
                pauliword[n] = pauli_type
        else:
            raise ValueError('invalid algorithm for constructing string operators')
            
        return ImagCoe_tot, pauliword
    
    def string_op(self, na:int, nb:int)->np.ndarray:
        """
        Generate the string operator of a single string, whose endpoints are na (in sublattice A) and nb (in sublattice B)
        """
        ImagCoe_tot, pauliword = self.string_op_pauliword(na=na, nb=nb)
        return  ImagCoe_tot * self.PauliString(pauliword=pauliword)
    
    def SigmaOp(self, setA:list, setB:list)->np.ndarray:
        """
        Producing the string operator of a general edge configuration, which may contain more than one string. setA consists of the type-A endpoints of the edge configuration, while setB contains the type-B ones. 
        """
        # A valid (good) edge configuration must contains the same amount of type-A endpoints and type-B endpoints
        if len(setA) != len(setB):
            raise ValueError('A valid (good) edge configuration must contains the same amount of type-A endpoints and type-B endpoints')
        
        # sort the sets such that the elements are arranged in an increasing order
        setA.sort()
        setB.sort()

        Sigma = sp.identity(self.dim, format="csr")   # initialize the string operator

        counting = 0
        for na in setA:
            nb = setB[counting]

            #print(na, nb)
            Sigma = Sigma @ self.string_op(na=na, nb=nb)

            counting += 1

        return Sigma
    
    ################### constructing rho ###################
    
    def single_plaquette_pauliword(self, site:int)->dict:
        """
        Return the Pauli word of a plaquette operator whose lower left corner is located at ``site``, which must be of type-B and is labeled by an integer
        """
        PlaqPauliword = {}

        movepoint = site    # The starting point, which is by default the lower left corner vertex of the rectangle (hence type-B), and circling counterclockwise 
        l, s = self.dictionary_n_to_l(label=movepoint)

        PlaqPauliword[movepoint] = 'X'
        
        s += 1
        movepoint = self.dictionary_l_to_n(layer=l,site=s)
        PlaqPauliword[movepoint] = 'Z'
        
        s += 1
        movepoint = self.dictionary_l_to_n(layer=l,site=s)
        PlaqPauliword[movepoint] = 'Y'
        
        l += 1
        s -= 1
        movepoint = self.dictionary_l_to_n(layer=l,site=s)
        PlaqPauliword[movepoint] = 'X'

        s -= 1
        movepoint = self.dictionary_l_to_n(layer=l,site=s)
        PlaqPauliword[movepoint] = 'Z'

        s -= 1
        movepoint = self.dictionary_l_to_n(layer=l,site=s)
        PlaqPauliword[movepoint] = 'Y'
        return  PlaqPauliword
    
    def plaquette_projector(self)->tuple[int, np.ndarray]:
        """
        The output is the number of plaquettes with the plaquette projector
        """

        ### sort lattice sites into A and B sublattices
        A_all = []
        B_all = []
        
        nA = 0
        nB = 0
        for i in np.arange(self.N, dtype=int):
            if i%2 == 0:
                B_all.append(i)
            else:
                A_all.append(i)

        PlaquetteProjector = sp.identity(self.dim, format='csr')    # initialize the plaquette projector
        
        p_number = 0
        
        for start in B_all:    # each starting point is the lower left corner vertex of the rectangle (hence type-B), and circling counterclockwise 
            PlaqPauliword = self.single_plaquette_pauliword(site=start)

            LocalPla = 0.5*(sp.identity(self.dim, format='csr') + self.PauliString(pauliword=PlaqPauliword))

            PlaquetteProjector = PlaquetteProjector @ LocalPla
            p_number += 1
        
        return p_number, PlaquetteProjector
    
    def wilson_loop(self, direction:str)->np.ndarray:
        """
        The variable direction is either 'horizontal' or 'vertical'. Inputs lying outside these two options are not allowed currently.
        """
        LoopPauliword = {}
        
        l, s = 0, 0   # moving point, which starts cricling the torus from the site n=0
        comeback = False
        if direction == 'horizontal':
            while not comeback:
                s += 1
                s = s%self.width    # PBC

                LoopPauliword[self.dictionary_l_to_n(layer=l, site=s)] = 'Z'
                
                if s%self.width == 0:
                    comeback = True
        elif direction == 'vertical':
            while not comeback:
                # moving to the upper left
                s -= 1
                l += 1
                s%self.width    # PBC
                l%self.height    # PBC

                LoopPauliword[self.dictionary_l_to_n(layer=l, site=s)] = 'Y'

                # moving to the right
                s += 1
                s%self.width    # PBC
                l%self.height    # PBC

                LoopPauliword[self.dictionary_l_to_n(layer=l, site=s)] = 'X'

                if s%self.width == 0:
                    comeback = True
        else:
            raise ValueError("The variable direction is either 'horizontal' or 'vertical'. Inputs lying outside these two options are not allowed currently.")
        
        return  self.PauliString(pauliword=LoopPauliword)
    
    def topo_projector(self)->np.ndarray:
        """
        0.25 * (I + Wh) @ (I + Wv) 
        """
        Wh, Wv = self.wilson_loop(direction='horizontal'), self.wilson_loop(direction='vertical')
        I = sp.identity(self.dim, format='csr')
        return  0.25 * (I + Wh) @ (I + Wv) 
    
    def density_matrix(self, update:bool=False)->np.ndarray:
        """
        Generate the density matrix (the entire system). 
        NOTE: the format of the output and saved density matrix is ``scipy.sparse``. To load the density matrix, please use ``sp.load_npz`` instead of ``np.load``
        """
        if update:  # recalculate the density matrix
            os.remove(self.datapath+'/'+'density-matrix.npz')

        if os.path.isfile(self.datapath+'/'+'density-matrix.npz'):
            return sp.load_npz(self.datapath+'/'+'density-matrix.npz')   
        else:
            config = self.EndpointSets()

            rho = None
            for D in range(self.num_total_config):
                setA, setB = config[D]

                Sigma = self.SigmaOp(setA=setA, setB=setB)
                #print('\033[34mProgress Report: Constructing string operators\033[0m')

                coe = self.Coefficient(num_config=D)
                #print('\033[34mProgress Report: Computing coefficients\033[0m')

                rho = coe * Sigma if rho is None else rho + coe * Sigma
                #print('\033[34mProgress Report: The %d-th configuration (size: %d sites) done\033[0m' % (D, 2*len(setA)))
            
            Np, p_proj = self.plaquette_projector()
            rho = 2 ** (-(self.N - Np)) * self.topo_projector() @  p_proj @ rho 
            self.rho = rho

            os.makedirs(name=self.datapath, exist_ok=True)
            sp.save_npz(self.datapath+'/'+'density-matrix.npz', rho)    # do not use np.save to save a sparse matrix
            
            return  rho

    ################### Translational Symmetry Trick ###################
    # Here, I would like to take advantage of translational symmetry and the nice property of the Pauli string.
    # Take the product spin states as the basis vectors, the value of <a|Sigma|b>, where |a> = 

    def state_to_spin(self, label:int)->list:
        """
        translate the n-th state to its spin configuration [s1, s2, ..., sN]
        # Convention
        0: up, 1: down

        0 -> [...,0,0,0]
        1 -> [...,0,0,1]
        2 -> [...,0,1,0]
        """
        bin_list = [int(x) for x in bin(label)[2:]]
        
        # fill in the missing spins (which are zeros)
        return [0 for n in range(int(self.N - len(bin_list)))] + bin_list
    
    def spin_to_state(self, config:list)->int:
        """
        translate a spin configuration [s1, s2, ..., sN] to its correaponding state label. 
        This is the inverse function of ``state_to_spin``
        # Convention
        0: up, 1: down

        [...,0,0,0] -> 0
        [...,0,0,1] -> 1
        [...,0,1,0] -> 2
        """
        return  int("".join(map(str, config)), 2)
    
    def translation(self, config:list, move:list[int,int])->list:
        """
        # Variables
        ``config``: the input spin configuration we want to translate
        ``move``: [x, y]: the vector of how far we want to translate (in the unit of unit cell, not lattice site)

        Define how a spin configuration are translated to another one
        Take a three-spin system with PBC for example, we have T([1,0,0]) = T([0,1,0])
        For a 4 by 4 lattice geometry, the labeling for the sites is
        [[12,   13, 14, 15]
        [8,     9,  10, 11]
        [4,     5,  6,  7]
        [0,     1,  2,  3]]
        RULE: left to right, lower to upper

        NOTE: The unit vector in the x-direction crosses TWO lattice sites
        """
        config_trans = config.copy()
        for l in range(self.height):
            for s in range(self.width):
                n = self.dictionary_l_to_n(layer=l, site=s) # the unit vector in the x-direction crosses TWO lattice sites
                config_trans[n] = config[self.dictionary_l_to_n(layer = l - move[1], site = s - 2*move[0])] # The factor 2 preceding ``move[0]`` is due to the fact that the unit vector in the x-direction crosses TWO lattice sites
        return  config_trans

    def translation_label(self, label:int, move:list[int, int])->int:
        '''
        Similar to the function `translation`, but the intput and output are labels rather than configurations
        '''
        config_trans = self.translation(config=self.state_to_spin(label=label), 
                                        move=move)
        return  self.spin_to_state(config=config_trans)
    
    def pauli_string_act(self, state:int, pauliword:dict)->tuple[np.complex64, int]:
        """
        # Output:
        (phase, state_label)

        Compute 
        
        PauliOp|state> = phase |state'>

        Convention: 0: up, 1: down
        """
        config = self,self.state_to_spin(label=state)
        
        phase = 1   # Pauli Y and Z gives an additional phase factor
        for i in range(self.N): # here, i is the label of the lattice sites
            if pauliword[i] == 'X':
                config[i] = (config[i]+1)%2 # spin flipping
            elif pauliword[i] == 'Y':
                phase = phase * (-1)**config[i] * 1j    # phase
                config[i] = (config[i]+1)%2 # spin flipping
            elif pauliword[i] == 'Z':
                phase = phase * (-1)**config[i] # phase
            else:
                pass
        return  phase, self.spin_to_state(config=config)
    
    def pauli_string_act_vec(self, basis:list, state_vec:np.ndarray, pauliword:dict)->np.ndarray:
        """
        Similar to the function ```pauli_string_act```, but the the input and the output are vectors represented by a specific set of ```basis```

        ``basis``:list = [n1, n2,...], where each ni is the state label

        ``state_vec``: ``np.ndarray`` = (v1, v2, ...)^T

        output: ``np.ndarray`` = (v'1, v'2, ...)^T

        op (v1, v2, ...)^T = (v'1, v'2, ...)^T
        """
        # check if the size of state_vec matches that of the basis
        if len(basis) != len(state_vec):
            raise ValueError('Because state_vec is represented in terms of basis vectors, the size of the given basis set must be the same as the size of the input vector')
        
        output_vec = np.zeros_like(state_vec)

        for i in range(len(basis)):
            # opn is the label of the output basis: 
            # OP |basis[i]> = coe |opn> = coe |basis[f]>
            coe, opn = self.pauli_string_act(state=basis[i], pauliword=pauliword) 

            if opn in basis:
                f = basis.index(opn)    # OP |basis[i]> = coe |opn> = coe |basis[f]>
            else:
                raise ValueError('OP |basis[i]> lies outside the basis set. The basis should be closed under this operation.')
                
            # OP v[i]|basis[i]> = coe v[i] |opn> = coe v[i] |basis[f]>
            output_vec[f] += coe * state_vec[i]

        return  output_vec
    
    def pauli_string_sandwitch(self, state_1:int, state_2:int, pauliword:dict)->np.complex64:
        """
        # Variables
        ``state_1``, ``state_2``: the state labels of the spin configurations
        ``paulistr``: [PauliX, PauliY, PauliZ], each set contains the sites where Pauli spin operators of that direction are located
        
        # Function
        Compute <state_1|Pauli_String_Op|state_2>
        """
        config_1, config_2 = self.state_to_spin(label=state_1), self.state_to_spin(label=state_2)
        
        value = 1
        for i in range(self.N): # here, i is the label of the lattice sites
            if i in pauliword:
                if pauliword[i] == 'X':
                    S = self.X
                elif pauliword[i] == 'Y':
                    S = self.Y
                elif pauliword[i] == 'Z':
                    S = self.Z
            else:
                S = self.I

            value = value * S[config_1[i], config_2[i]]
        
        return  value
    
    def representative_set(self, update:bool=False)->list:
        '''
        A list of all the representative states
        '''
        if update and os.path.isfile("w_%d_h_%d/representative_set.json" % (self.width, self.height)):
            os.remove("w_%d_h_%d/representative_set.json" % (self.width, self.height))

        if os.path.isfile("w_%d_h_%d/representative_set.json" % (self.width, self.height)):
            with open("w_%d_h_%d/representative_set.json" % (self.width, self.height), "r") as file:
                rep_list = json.load(file)
        else:
            state_list = list(range(self.dim))
            rep_list = []

            find_next_rep = True
            while find_next_rep:
                rep_state = state_list[0]
                rep_list.append(rep_state) # start with the first element of the state list at each iteration step
                for x in range(int(self.width/2)): # the unit vector in the x-direction crosses TWO lattice sites
                    for y in range(self.height):
                        nt = self.spin_to_state(config=self.translation(config=self.state_to_spin(label=rep_state), move=[x,y]))
                        if nt in state_list:
                            #print('remove %d' % nt)
                            if nt == 65535:
                                print('the representative that is equivalent to 65535 is %d' % rep_state)
                            state_list.remove(nt)

                if state_list == []:
                    find_next_rep = False

            os.makedirs("w_%d_h_%d" % (self.width, self.height), exist_ok=True)
            with open("w_%d_h_%d/representative_set.json" % (self.width, self.height), "w") as file:
                json.dump(rep_list, file)
        return  rep_list
    
    def period_dict(self, update:bool=False)->dict:
        """
        A period R is the smallest integer such that T^R |state> = |state>
        
        # Output:
        {
            "rep1": [Rx_1, Ry_1]
            "rep2": [Rx_2, Ry_2]
            ...
        }
        where each ``repi`` is an integer
        """
        if update:
            os.remove("w_%d_h_%d/period_dict.json" % (self.width, self.height))

        if os.path.isfile("w_%d_h_%d/period_dict.json" % (self.width, self.height)):
            with open("w_%d_h_%d/period_dict.json" % (self.width, self.height), "r") as file:
                period_dic = json.load(file)
        else:
            rep_list = self.representative_set()
            period_dic = {}
            for n in rep_list:
                ######## search the period in x
                x_search = True
                Rx = 1  # period R=0 trivially satisfies T^R = 1
                while x_search:
                    nt = self.spin_to_state(config=self.translation(config=self.state_to_spin(label=n), move=[Rx,0]))
                    if nt != n:
                        Rx += 1
                    else:
                        x_search = False
                ######## search the period in y
                y_search = True
                Ry = 1
                while y_search:
                    nt = self.spin_to_state(config=self.translation(config=self.state_to_spin(label=n), move=[0,Ry]))
                    if nt != n:
                        Ry += 1
                    else:
                        y_search = False
                period_dic["%d" % n] = [Rx, Ry]
            with open("w_%d_h_%d/period_dict.json" % (self.width, self.height), 'w') as file:
                json.dump(period_dic, file)
        return  period_dic
    
    def momentum_sectors(self, update:bool=False)->dict:
        """
        # Output:
        {
            "(m1_x,m1_y)": [rep1_k1, rep2_k1, ...]_k1, 
            "(m2_x,m2_y)": [rep1_k2, rep2_k2, ...]_k2,
            ... 
        }

        First we have to define the periodicity R of a state psi, which is the smallest integer such that T^R |psi> = |psi>
        An allowed representative in the momentum sector k must satisfy exp(ikR) = 1. Otherwise, the corresponding momentum state generated from such a representative will vanish.
        Each sector represents a diagonal block, with its corresponding momentum, of the density matrix
        
        The momentum states we consider here are
        m_mu = 0, 1, 2, ..., N_mu-1
        k_mu = m_mu * 2pi/N_mu
        """
        if update:
            os.remove("w_%d_h_%d/momentum_sectors.json" % (self.width, self.height))

        if os.path.isfile("w_%d_h_%d/momentum_sectors.json" % (self.width, self.height)):
            with open("w_%d_h_%d/momentum_sectors.json" % (self.width, self.height), "r") as file:
                momentum_sectors_dict = json.load(file)
        else:
            rep_list = self.representative_set()
            period_dict = self.period_dict()

            momentum_sectors_dict = {}
            for mx in range(int(self.width/2)):  # the unit vector in the x-direction crosses TWO lattice sites
                for my in range(self.height):
                    
                    k_sector = []   #   start collecting representatives into the momentum sector
                    for n in rep_list:
                        [Rx, Ry] = period_dict["%d" % n]
                        #print("Rx=%d, Ry=%d, mx=%d, my=%d" % (Rx, Ry, mx, my))
                        if mx*Rx % (self.width/2) == 0 and my*Ry % self.height == 0:    # meaning that exp(ikR) = 1
                            k_sector.append(n)
                    momentum_sectors_dict["(%d,%d)" % (mx, my)] = k_sector
            with open("w_%d_h_%d/momentum_sectors.json" % (self.width, self.height), "w") as file:
                json.dump(momentum_sectors_dict, file)
        
        return  momentum_sectors_dict

    def diagonal_block(self, basis:list, op:dict, **kwargs)->np.ndarray:
        """
        # Variables
        ## ``basis``: list
        The variable ``basis`` is a list of state labels

        basis = [n1, n2, ...]

        ## ``op``: dict
        Operators are expanded in terms of Pauli strings and are stored in the following form:
        {
            "component0": (coe_0, pauliword_0)
            "component1": (coe_1, pauliword_1)
            "component2": (coe_1, pauliword_2)
            ...
        }
        The mathmatical expression of the operator is then

        op = sum_{i} coe_i * PauliStr

        ## ``**kwargs``
        - hermitian
        - symmetric

        # Output:
        [block]_{n1,n2} = <n1| op |n2> = sum_{i} coe_i * <n1| PauliStr |n2>
        """
        ishermitian, issymmetric = False, False
        if 'hermitian' in kwargs:
            ishermitian = kwargs['hermitian']
        if 'symmetric' in kwargs:
            issymmetric = kwargs['symmetric']

        block_mat = np.zeros([len(basis), len(basis)], dtype=np.complex64)
        
        for cpnt in range(len(op)):    # the number of components of the operator expanded in terms of Pauli strings
            coe, pauliword = op['component%d'%cpnt]  # read the information of the current component
            
            # construct the matrix of the current component
            cpnt_mat = np.zeros_like(block_mat, dtype=np.complex64)
            for n1 in basis:
                for n2 in basis:
                    if ishermitian:
                        if n2 < n1:
                            cpnt_mat[n1,n2] = cpnt_mat[n2,n1].conjugate()
                            continue
                    elif issymmetric:
                        if n2 < n1:
                            cpnt_mat[n1,n2] = cpnt_mat[n2,n1]
                            continue
                    else:
                        pass
                    cpnt_mat[n1,n2] = coe * self.pauli_string_sandwitch(state_1=n1, 
                                                                         state_2=n2,
                                                                         pauliword=pauliword)
            # add the current component to the total matrix
            block_mat = block_mat + cpnt_mat
        return  block_mat
    
    def diagonal_block_LinearOperator(self, basis:list, op:dict)->LinearOperator:
        """
        This function does the same thing as the function ``diagonal_block``, but the output is a LinearOperator

        # Variables
        ## ``basis``: list
        The variable ``basis`` is a list of state labels

        basis = [n1, n2, ...]

        ## ``op``: dict
        Operators are expanded in terms of Pauli strings and are stored in the following form:
        {
            "component1": (coe_1, pauliword_1)
            "component2": (coe_2, pauliword_2)
            ...
        }
        The mathmatical expression of the operator is then

        op = sum_{i} coe_i * PauliStr

        ## ``**kwargs``
        - 'hermitian'
        - 'symmetric'

        # Output:
        [block]_{n1,n2} = <n1| op |n2> = sum_{i} coe_i * <n1| PauliStr |n2>
        """
        block_mat = None
        for cpnt in len(op):    # the number of components of the operator expanded in terms of Pauli strings
            coe, pauliword= op['component%d'%cpnt]  # read the information of the current component (cpnt)
            
            # define the matvec function of the current component
            # vec = v1*basis1 + v2*basis2 + ... = (v1, v2, ...)
            matvec_cpnt = lambda vec: coe * self.pauli_string_act_vec(basis=basis,
                                                                      state_vec=vec,
                                                                      pauliword=pauliword)
            block_mat = LinearOperator((len(basis), len(basis)), matvec=matvec_cpnt, dtype=np.complex64) if block_mat is None else block_mat + LinearOperator((len(basis), len(basis)), matvec=matvec_cpnt, dtype=np.complex64)

        return  block_mat
    
    def SigmaOp_pauliword(self, setA:list, setB:list)->tuple[np.complex64, dict]:
        """
        Producing the pauliword of the string operator of a general edge configuration, which may contain more than one string. setA consists of the type-A endpoints of the edge configuration, while setB contains the type-B ones. 
        
        SigmaOp = ImagCoe * PauliString = stringop1 @ stringop2 @ stringop3....
        """
        # A valid (good) edge configuration must contains the same amount of type-A endpoints and type-B endpoints
        if len(setA) != len(setB):
            raise ValueError('A valid (good) edge configuration must contains the same amount of type-A endpoints and type-B endpoints')
        
        # sort the sets such that the elements are arranged in an increasing order
        setA.sort()
        setB.sort()

        ImagCoe_tot = 1 # initialize the total phase factor ImagCoe
        SigmaPauliWord_tot = None   # initialize the pauliword of the output Pauli string operator

        counting = 0    # count for the setB
        for na in setA:
            nb = setB[counting]

            imcoe, stringpauliword = self.string_op_pauliword(na=na, nb=nb)

            ImagCoe_tot = ImagCoe_tot * imcoe
            SigmaPauliWord_tot = stringpauliword if SigmaPauliWord_tot is None else pauliwords_mult(pauliword1=SigmaPauliWord_tot, pauliword2=stringpauliword)

            counting += 1

        return ImagCoe_tot, SigmaPauliWord_tot
    
    def plaquette_projector_momentum_block(self, k:list[int,int], Z2sym:bool=True)->tuple[int, np.ndarray]:
        """
        The output is the number of plaquettes with the plaquette projector
        """

        ### sort lattice sites into A and B sublattices
        A_all = []
        B_all = []
        
        nA = 0
        nB = 0
        for i in np.arange(self.N, dtype=int):
            if i%2 == 0:
                B_all.append(i)
            else:
                A_all.append(i)

        k_basis_set = self.momentum_sectors()['(%d,%d)'%(k[0],k[1])]    # give the basis sector
        
        # initialize the plaquette projector, which is the identity of the momentum sector
        PlaquetteProjector = sp.identity(len(k_basis_set))    
        
        p_number = 0
        for start in B_all:    # each starting point is the lower left corner vertex of the rectangle (hence type-B), and circling counterclockwise 
            PlaqPauliword = self.single_plaquette_pauliword(site=start)

            LocalPla = 0.5*(sp.identity(self.dim, format='csr') + self.diagonal_block(basis=k_basis_set,
                                                                                      op={"component0":(1, PlaqPauliword)},
                                                                                      hermitian=True))

            PlaquetteProjector = PlaquetteProjector @ LocalPla
            p_number += 1
        
        return p_number, PlaquetteProjector
    
    def density_matrix_momentum_block(self, k:list[int,int], Z2sym:bool=True, update:bool=False)->np.ndarray:
        """
        ``k``: momentum = [kx, ky]

        ``Z2sym``: whether the system possesses a Z2 symmetry
        """

        isZ2_filename = '_Z2sym' if Z2sym else ''

        pathname = self.datapath+'/'+'density-matrix_momentum-block' + isZ2_filename 
        os.makedirs(name=pathname, exist_ok=True)

        save_filename = pathname + '/' + 'kx_%d_ky_%d' % (k[0], k[1])

        if update:  # recalculate the density matrix
            os.remove(save_filename +'.npy')

        if os.path.isfile(save_filename +'.npy'):
            return np.load(save_filename +'.npy')   
        else:
            k_basis_set = self.momentum_sectors()['(%d,%d)'%(k[0],k[1])]    # give the basis sector

            config = self.EndpointSets()

            rho = None
            for D in range(self.num_total_config):    # looping all the possible configurations
                setA, setB = config[D]

                Sigma = self.SigmaOp(setA=setA, setB=setB)
                #print('\033[34mProgress Report: Constructing string operators\033[0m')

                coe = self.Coefficient(num_config=D)
                #print('\033[34mProgress Report: Computing coefficients\033[0m')

                rho = coe * Sigma if rho is None else rho + coe * Sigma
                #print('\033[34mProgress Report: The %d-th configuration (size: %d sites) done\033[0m' % (D, 2*len(setA)))
            
            Np, p_proj = self.plaquette_projector()
            rho = 2 ** (-(self.N - Np)) * self.topo_projector() @  p_proj @ rho 
            self.rho = rho

            os.makedirs(name=self.datapath, exist_ok=True)
            sp.save_npz(self.datapath+'/'+'density-matrix.npz', rho)    # do not use np.save to save a sparse matrix
            
            return  rho

    ################### quantum magic ###################

    def norm_cond_test(self, approximate:bool=False, 
                            update:bool=False,
                            lazy:bool = False)->float:
        # if approximate:
        #     if lazy:
        #         return 2 * sum(np.array(list(self.calculate_CoefficientTable_approx(update=update, lazy=True).values())) ** 2) / 2**(self.N/2)  # The factor 2 is due to the approximation
        #     else:
        #         return sum(np.array(list(self.calculate_CoefficientTable_approx(update=update).values())) ** 2) / 2**(self.N/2)  
        # else:
        coe_info = np.array(list(self.calculate_CoefficientTable(update=update).values()))
        coe_array = coe_info[:,1]

        return sum(coe_array**2) / 2**(self.N/2)

    def stab_renyi_entropy(self, n:int, 
                                approximate:bool=False, 
                                update:bool=False,
                                lazy:bool = False)->np.ndarray:
        """
        Compute the n-stabilizer Renyi entropy of the Kitaev ground state
        M_n = (1-n)^{-1} ln( 2^{-N/2} * sum_{[D]} C_D^{2*n})
        """  
        coe_info = np.array(list(self.calculate_CoefficientTable(update=update).values()))
        coe_array = coe_info[:,1]
        if n == 1:
            coe_array_finite = np.array([n for n in coe_array if n != 0])
            return - float(coe_array_finite**2  @ np.log(coe_array_finite**2)) / 2**(self.N/2) 
        else:
            return (1-n)**(-1) * np.log( np.sum(coe_array**(2*n)) / 2**(self.N/2) )

    def RDM_2_stab_renyi_entropy(self, tot_size:list[int, int]=None, 
                                 thermo_limit:bool=False, 
                                 update=False,
                                 corr_table_sys_size:list[int,int]=None):
        '''
        ``tot_size`` = [the width of the entire system, the height of ..]
        If ``thermo_limit`` is true, the total system size is set to be infinite, and the correlation function is generated by the integration function.
        The defualt value of ``thermo_limit`` is true, which means that the output is the data for an infinitely large system if one doesn't set ``tot_size``
        '''
        self.thermo_limit = thermo_limit    # note that if termo_limit is true, then the tot_size won't be taken.

        if tot_size is None and thermo_limit==False:
            raise KeyError('Please set the variable ``tot_size`` or make ``thermo_limit`` true. Otherwise, please use the function stab_renyi_entropy for calculating the pure state SRE.')
        
        # self.full_corr_mat_update(tot_size=tot_size)
        print('unpack the dictionary')
        coe_info = np.array(list(self.calculate_CoefficientTable(update=update, 
                                                                 tot_size=tot_size,
                                                                 corr_table_sys_size=corr_table_sys_size).values()))
        coe_array = coe_info[:,1]
        # print(coe_array)
        if self.full_corr_mat is None:
            self.full_corr_mat_update(corr_table_sys_size=corr_table_sys_size)
        denominator = la.det(np.eye(self.N) + self.full_corr_mat)
        # denominator = np.sum(coe_array**2)
        print('compute the SRE')
        enumerator = np.sum(coe_array**4)
        
        return -np.log(enumerator / denominator)

    def SRE_1D(self, Jx:float=None,  Jy:float=None,
               tot_length:int=None, thermo_limit:bool=False, update:bool=False):
        """
        Calculate the 2-SRE of a 1D Kitaev system. 
        NOTE: The subsystem SRE calculated by this function is NOT deducted by the Renyi entropy of the subsystem.
        
        ``tot_length``: the length of the entire spin chain
        ``thermo_limit``: set to be ``True`` if the length of the entire spin chain is infinity
        ``corr_table_sys_size``: the system size of the corresponding correlation table that will be used to calculate the coefficient
        ``update``: update the corrlation table or not.
        """
        Jx_old, Jy_old, Jz_old = self.Jx, self.Jy, self.Jz

        self.Jz = 0 # decoupling the chains
        if not Jx is None:
            self.Jx = Jx
        if not Jy is None:
            self.Jy = Jy

        self.height = 1 # this function consider only the spin chain

        if tot_length is None and thermo_limit==False:
            raise KeyError('Please set the variable ``tot_length`` or make ``thermo_limit`` true. Otherwise, please use the function stab_renyi_entropy for calculating the pure state SRE.')

        # self.full_corr_mat_update(tot_size=tot_size)
        coe_info = np.array(list(self.calculate_CoefficientTable(update=update, 
                                                                 tot_size=[tot_length, 2],
                                                                corr_table_sys_size=[tot_length, 1]).values()))
        coe_array = coe_info[:,1]

        # zero_point_entropy = 0
        # if self.width == tot_length:    # calculating the entropy of the pure state (the entire system)
        #     if self.width%4 == 0:   # k=1/2 hits the energy-zero point, contributing an entropy ln2
        #         zero_point_entropy = np.log(2)

        self.Jx, self.Jy, self.Jz = Jx_old, Jy_old, Jz_old  # restore the coupling strength
        
        return -np.log(np.sum(coe_array**4) / 2**(self.N/2)) # - zero_point_entropy

    def mutual_SRE_1D(self, Jx:float,  Jy:float,
                      L:int, l:int, update:bool=False)->float:
        """
        Calculating the mutual stabilizer renyi entropy of a Kitaev chain.
        ``L`` is the length of the entire system (PBC), and ``l | (L-l)`` is the division of the system

        The definition of the mutual SRE reads
        ```
        SRE_l + SRE_env - SRE_loop
        ```
        where ``SRE_l`` is the subsystem SRE of length ``l``
        ``SRE_env`` is the subsystem SRE of length ``(L - l)``
        """
        Jx_old, Jy_old, Jz_old = self.Jx, self.Jy, self.Jz

        self.Jz = 0 # decoupling the chains
        self.Jx, self.Jy = Jx, Jy
        
        if L%2==1:
            raise ValueError('L, the length along the xy-direction, must be even to agree with the periodic boundary condition')
        if L < 4:
            raise ValueError('L is too small. It must be larger than 4')
        
        self.width, self.height = l, 1
        SRE_l = self.SRE_1D(tot_length=L,  update=update)
        
        if l%2 == 1:
            self.ini_lat_type = 'A'
        self.width, self.height = L-l, 1
        SRE_env = self.SRE_1D(tot_length=L,  update=update)
        
        if l%2 == 1:    # restore the setting
            self.ini_lat_type = 'B'    
        self.width, self.height = L, 1
        SRE_loop = self.SRE_1D(tot_length=L,  update=update)
        
        print(f'SRE_l={SRE_l}, SRE_env={SRE_env}, SRE_loop={SRE_loop}')

        self.Jx, self.Jy, self.Jz = Jx_old, Jy_old, Jz_old  # restore the coupling strength

        return SRE_l + SRE_env - SRE_loop

    def renyi_entropy_1D(self, Jx:float, Jy:float, 
                            tot_length:int=None, thermo_limit:bool=False, 
                            corr_table_sys_size:list[int,int]=None, update:bool=False):
        """
        This function calculates the 2-Renyi entropy of a Kitaev chain (xy-alterating chain, not the Kitaev spin chain model).  
        ``tot_length``: the length of the entire spin chain
        ``thermo_limit``: set to be ``True`` if the length of the entire spin chain is infinity
        ``corr_table_sys_size``: the system size of the corresponding correlation table that will be used to calculate the coefficient
        ``update``: update the corrlation table or not.
        """
        self.Jz = 0 # decoupling the chains
        self.Jx, self.Jy = Jx, Jy

        if self.height != 1:
            raise ValueError('This function only calculates the entropy of linear regions that the height must be 1')
        if tot_length is None and thermo_limit==False:
            raise KeyError('Please set the variable ``tot_length`` or make ``thermo_limit`` true. Otherwise, please use the function stab_renyi_entropy for calculating the pure state SRE.')
        
        self.full_corr_mat_update(tot_size=[tot_length, 2], # the height can actually be arbitrary, since the chains are decoupled 
                                  corr_table_sys_size=corr_table_sys_size, 
                                  update=update)
        entropy = -np.log(la.det(np.eye(self.N) + self.full_corr_mat) / (2**(self.N / 2)))

        zero_point_entropy = 0
        if self.width == tot_length:    # calculating the entropy of the pure state (the entire system)
            if self.width%4 == 0:   # k=1/2 hits the energy-zero point, contributing an entropy ln2
                zero_point_entropy = np.log(2)

        entropy = entropy - zero_point_entropy

        print(f'L={self.width}, Renyi entropy={entropy}')
        
        return entropy

################################## extra functions ##################################

def PauliMultRule(type1:str, type2:str)->tuple[np.complex64, str]:
    '''
    RULE: the Pauli type must be capital
    \sigma^type1 \sigma^type2 = ImagCoe \sigma^ResultType
    '''
    rule = {
        'XY': (1j, 'Z'),    'YX': (-1j, 'Z'),
        'YZ': (1j, 'X'),    'ZY': (-1j, 'X'),
        'ZX': (1j, 'Y'),    'XZ': (-1j, 'Y'),
        'idX': (1, 'X'),    'Xid': (1, 'X'),
        'idY': (1, 'Y'),    'Yid': (1, 'Y'),
        'idZ': (1, 'Z'),    'Zid': (1, 'Z'),
        'XX': (1, 'id'),
        'YY': (1, 'id'),
        'ZZ': (1, 'id'),
        'idid': (1, 'id')
        }
    comb_str = type1 + type2
    return  rule[comb_str]
        
def pauliwords_mult(pauliword1:dict, pauliword2:dict)->tuple[np.complex64, dict]:
    """
    Multiply two Pauli strings: 
    
    pualistr1*paulistr2 = ImagCoe * paulistr_output

    The output is the Pauli word corresponding to the resulting Pauli string ``paulistr_output``
    """
    outputPauliword = {}
    ImagCoe_total = 1
    union = pauliword1 | pauliword2
    for site in union:
        if site in pauliword1 and site in pauliword2:
            imcoe, new_type = PauliMultRule(type1=pauliword1[site], type2=pauliword2[site])
            ImagCoe_total = ImagCoe_total * imcoe
            outputPauliword[site] = new_type
        else:
            outputPauliword[site] = union[site]
    return  ImagCoe_total, outputPauliword

def read_num(input:str, prompt:str)->int:
    """
    Find the number after the prompt

    input = 'prompt num'
    """
    num_list = ''

    searching = True
    l = 0
    while searching:
        tobetest = ''
        for k in range(len(prompt)):
            tobetest += input[l+k]
            num_start_point = l+k+1 # the place of the first digit

        if tobetest == prompt:
            finddigit = True
            cursor = 0
            while finddigit:
                if num_start_point + cursor + 1 > len(input):
                    finddigit = False   # terminate the searching 
                    searching = False
                    break

                if input[num_start_point + cursor].isdigit():
                    num_list += input[num_start_point + cursor]
                    cursor += 1
                else:
                    finddigit = False   # terminate the searching 
                    searching = False
        l += 1
    return  int(num_list)

############# labeling endpoint sets (written using Claude, not by my own)

def _comb_rank(n, k, combo0):
    """
    Rank (0-indexed) of a k-subset 'combo0' (sorted, 0-indexed positions
    from {0,...,n-1}) within the lexicographic ordering produced by
    itertools.combinations(range(n), k).
    """
    rank = 0
    prev = -1
    for i, c in enumerate(combo0):
        for j in range(prev + 1, c):
            rank += comb(n - 1 - j, k - i - 1)
        prev = c
    return rank

def _comb_unrank(n, k, rank):
    """Inverse of _comb_rank: given a 0-indexed rank, return the sorted
    0-indexed k-subset (as positions in {0,...,n-1})."""
    result = []
    x = 0
    for i in range(k):
        while True:
            c = comb(n - 1 - x, k - i - 1)
            if rank < c:
                result.append(x)
                x += 1
                break
            else:
                rank -= c
                x += 1
    return result

def build_universes(N):
    """
    N is the TOTAL number of elements across both sets; each set has
    M = N/2 elements.
    Universe 1: odd integers  1, 3, 5, ..., N-1
    Universe 2: even integers 0, 2, 4, ..., N-2
    e.g. N=4 -> U1=[1,3], U2=[0,2]
    """
    if N % 2 != 0:
        raise ValueError("N must be even (it's the total element count "
                          "across both equally-sized sets)")
    U1 = list(range(1, N, 2))   # odd:  1, 3, ..., N-1
    U2 = list(range(0, N, 2))   # even: 0, 2, ..., N-2
    assert len(U1) == len(U2) == N // 2, (U1, U2)
    return U1, U2

def config_to_number(N, S1, S2, one_indexed=True):
    """
    U1 = odd integers 1..N-1, U2 = even integers 0..N-2 (each of size
    M = N/2). S1 must be a subset of U1, S2 a subset of U2, both the
    same size k. Returns the position of (S1, S2) in the enumeration
    order:
        k=0: [[],[]]
        k=1: [[U1[0]],[U2[0]]], [[U1[0]],[U2[1]]], ..., [[U1[1]],[U2[0]]], ...
        ...
        k=M: [[all of U1],[all of U2]]
    """
    U1, U2 = build_universes(N)
    M = len(U1)
    pos1 = {v: i for i, v in enumerate(U1)}
    pos2 = {v: i for i, v in enumerate(U2)}

    S1, S2 = list(S1), list(S2)
    k = len(S1)
    if len(S2) != k:
        raise ValueError("S1 and S2 must have the same size")
    if len(set(S1)) != k or len(set(S2)) != k:
        raise ValueError("S1 and S2 must have no repeated elements")
    if any(x not in pos1 for x in S1):
        raise ValueError(f"S1 elements must come from the odd universe {U1}")
    if any(x not in pos2 for x in S2):
        raise ValueError(f"S2 elements must come from the even universe {U2}")

    # how many pairs precede this k-block
    offset = sum(comb(M, j) ** 2 for j in range(k))

    # rank of S1 within U1, and S2 within U2 (map elements -> positions first)
    r1 = _comb_rank(M, k, sorted(pos1[x] for x in S1))
    r2 = _comb_rank(M, k, sorted(pos2[x] for x in S2))

    idx = offset + r1 * comb(M, k) + r2
    return idx + 1 if one_indexed else idx

def number_to_config(N, number, one_indexed=True):
    """Inverse: given the position number, return (S1, S2) using the
    actual odd/even values from U1, U2."""
    U1, U2 = build_universes(N)
    M = len(U1)

    idx = number - 1 if one_indexed else number
    k = 0
    while True:
        block = comb(M, k) ** 2
        if idx < block:
            break
        idx -= block
        k += 1

    ck = comb(M, k)
    r1, r2 = divmod(idx, ck)
    S1 = [U1[i] for i in _comb_unrank(M, k, r1)]
    S2 = [U2[i] for i in _comb_unrank(M, k, r2)]
    return S1, S2

def ABsort(endpointset:list)->tuple[list, list]:
    setA = [n for n in endpointset if n%2==1]
    setB = [n for n in endpointset if n%2==0]
    return setA, setB

def label_to_endpointset(label:int)->tuple[list, list]:
    '''
    `label` is the decimal rep of endpoint config. This function converts the decimal label to the corresponding endpoint set.
    The output is the endpoint set divided into two sublattices (setA, setB)
    '''
    bi_string_list = list(np.binary_repr(label))
    endpoints = [n for n in range(len(bi_string_list)) if bi_string_list[-n-1]=='1']
    return ABsort(endpointset=endpoints)

def find_key(dictionary:dict, target)->str:
    """
    Find the parent key of the target value.
    """
    return  next((k for k, v in dictionary.items() if v == target), None)

# @njit(parallel=True)
def corr_pfaffian_calculator(mat:np.ndarray, endpointset:np.ndarray)->float:
    '''
    Calculate the pfaffian 
    '''
    if len(endpointset) == 0:
        cor_pfa = 1 # vacuum configuration
    else:
        cor_mat = np.zeros((len(endpointset), len(endpointset)))
        for i in np.arange(len(endpointset)):
            for j in np.arange(i, len(endpointset)):
                if i == j:
                    cor_mat[i,i] = 0
                else:
                    cor_mat[i,j] = mat[endpointset[i], endpointset[j]]
                    cor_mat[j,i] = - cor_mat[i,j]
        cor_pfa = pf.pfaffian(cor_mat)
    return cor_pfa

def derivative_1st(x_data:list, y_data:list)->tuple[np.ndarray,np.ndarray]:
    x_len, y_len = len(x_data), len(y_data)
    if x_len != y_len:
        raise ValueError('The sizes of the two arrays do not match.')
    x_output, y_output = np.zeros(int(x_len - 1)), np.zeros(int(x_len - 1))
    for n in range(x_len-1):
        x_output[n] = (x_data[n+1] + x_data[n])/2
        y_output[n] = (y_data[n+1] - y_data[n]) / (x_data[n+1] - x_data[n])
    return x_output, y_output

def derivative_2nd(x_data:list, y_data:list)->tuple[np.ndarray,np.ndarray]:
    x_len, y_len = len(x_data), len(y_data)
    if x_len != y_len:
        raise ValueError('The sizes of the two arrays do not match.')
    x_data_1st, y_data_1st = derivative_1st(x_data=x_data, y_data=y_data)   # first derivative
    return derivative_1st(x_data=x_data_1st, y_data=y_data_1st)   # second derivative