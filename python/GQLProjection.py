import numpy as np
from dedalus.core.future import FutureField
from dedalus.core.operators import NonlinearOperator, parseable

@parseable('Project')
class GQLProjection(NonlinearOperator, FutureField):
    """
    Projection operator for generalized quasilinear approximation
    
    
    """
    def __init__(self, arg, cutoff, subspace, dim=None,**kw):
        super().__init__(arg,**kw)
        self.cutoff = cutoff
        self.tensorsig = self.args[0].tensorsig
        self.domain = self.args[0].domain
        self.dist   = self.args[0].dist
        self.dtype = self.args[0].dtype
        self.layout = self.dist.coeff_layout
        # by default, execute GQL on all but the last dimension
        if not dim:
            self.dim = self.domain.dim - 1
        else:
            self.dim = dim
        
        local_coeff = self.layout.local_group_arrays(self.domain, self.args[0].scales)
        low_mask = np.ones(local_coeff[0].shape, dtype='bool')

        for i in range(self.dim):
            low_mask &= (np.abs(local_coeff[i]) <= cutoff[i])
        if subspace == 'high' or subspace == 'h':
            self.mask = ~low_mask
            subspace_name = 'h'
        elif subspace == 'low' or subspace == 'l':
            self.mask = low_mask
            subspace_name = 'l'
        else:
            raise ValueError("Subspace must be high/h or low/l, not {}".format(subspace))
        # expand on tensorsig
        self.mask = np.expand_dims(self.mask, axis=tuple(i for i in range(len(self.tensorsig))))
        cutoff_str = ",".join([str(i) for i in cutoff]) 
        self.name = 'Proj[({}),{}]'.format(cutoff_str,subspace_name)

    def meta_constant(self, axis):
        # Preserve constancy
        return self.args[0].meta[axis]['constant']

    def check_conditions(self):
        """Projection must be in coefficient space""" 
        return self.args[0].layout is self._coeff_layout
    
    def enforce_conditions(self):
        self.args[0].require_coeff_space()
    
    def operate(self, out):
        #for i in range(self.dim):
        #    self.args[0].require_layout('c')
        out.preset_layout(self.layout)
        out.data[:] = self.args[0].data
        out.data *= self.mask
