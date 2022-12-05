from radis import *
import numpy as np
from scipy.spatial import distance
from scipy.stats import wasserstein_distance


def Error_function(data_norm_store, reconstructed_observation,estimated_state):
    error_y=np.log(distance.euclidean(data_norm_store.orig_observation,reconstructed_observation)+1)
    # error_y = distance.euclidean(data_norm_store.actual_observation, reconstructed_observation)
    # error_cos=distance.cosine(actual_observation,reconstructed_observation)
    # error_y = distance.euclidean(actual_observation, reconstructed_observation)
    # error_y=wasserstein_distance(actual_observation,reconstructed_observation)*10.
    # error_reg=np.sum(np.maximum(estimated_state-1,np.zeros_like(estimated_state))+np.maximum(0-estimated_state,np.zeros_like(estimated_state)))
    error_reg_0=np.exp(np.maximum(estimated_state[0]-data_norm_store.feasible_domain[1,0],0)+
                       np.maximum(data_norm_store.feasible_domain[0,0]-estimated_state[0],0))-1
    error_reg_1 = np.exp(np.maximum(estimated_state[1] - data_norm_store.feasible_domain[1,1], 0) + np.maximum(data_norm_store.feasible_domain[0,1] - estimated_state[1], 0))-1
    error_group=np.array([error_y,error_reg_0,error_reg_1])*data_norm_store.error_zoomer
    error_sum=np.sum(error_group)
    return error_y*data_norm_store.error_zoomer,error_sum


class forward_model_radis:
    """
    this module is the physical forward model used for spectrum calculation
    """

    def __init__(self, molecule, waveneed, lightpath, stepwise=0.01,
                 databank='hitemp', spec_type='absorbance'):
        """
        INPUT:
        input2physics: the state/input, i.e., temp and mole fraction, which is an numpy array here, to generate the spectrum
        molecule: the molecule types, can be "CO2","H2O", etc. Be careful, must be CAPITAL WRITING
        waveneed: the waveband used to calculate spectrum unit: cm-1
        lightpath: the length of light path, with the unit of cm
        stepwise: the interval between every two wavenumber
        databank: the dababase used to calculate spectrum, options:'hitemp','hitran', etc. check radis
        spec_type: current can be used includes "absorbance', "absorptivity",'transmittance_noslit',"radiance_noslit", etc.
                the case "absorptivity" is a homemade one, which is used to calculate absorptivity/absorption spectroscopy,
                which is the showcase in the paper
                the case "radiance_noslit" is used to calculate emission spectroscopy, another showcase of the paper
        -----------------------------------------------------------------------
        OUTPUT:
        spectrum_generated: as the name indicates, the type has been transformed to np.array
        """
        self.molecule = molecule
        self.wave_need = waveneed
        self.wave_min = waveneed[0] - 1
        self.wave_max = waveneed[-1] + 1
        self.lightpath = lightpath
        self.databank = databank
        self.spec_type = spec_type
        self.stepwise = stepwise

    def generate_sample(self, state):
        s = calc_spectrum(self.wave_min, self.wave_max,  # cm-1
                          molecule=self.molecule,
                          isotope='1',
                          pressure=1.01325,  # bar
                          wstep=self.stepwise,
                          Tgas=state[0].item(),  # K
                          mole_fraction=state[1].item(),
                          path_length=self.lightpath,  # cm
                          databank=self.databank,  # or 'hitemp'
                          warnings={'AccuracyError': 'ignore'}
                          )
        # s.apply_slit(0.5, 'nm')       # simulate an experimental slit
        if self.spec_type == "absorptivity":
            wave, spectrum_generated = s.get('transmittance_noslit', wunit='cm-1')
            spectrum_generated = np.interp(self.wave_need, wave, 1 - spectrum_generated)
        else:
            wave, spectrum_generated = s.get(self.spec_type, wunit='cm-1')
            spectrum_generated = np.interp(self.wave_need, wave, spectrum_generated)
        # spectrum = np.interp(self.wave_need, wave, spectrum)
        return np.array(spectrum_generated)

def physics_evaluation_module(state_norm, data_norm_store, forward_physical_model):
    """
    this function is used to calculate error according to the given state
    ---------------------------------------------------------------------
    INPUT:
    state_norm: normalized state (to feasible domain)
    data_norm_store: the normalization_data_store class in data_operation.py, which contains all necessary and basic information
        of the problem
    forward_physical_model: the forward physical model created here
    ---------------------------------------------------------------
    OUTPUT:
    as name shows, no need further explain
    """
    if state_norm[0].item()>3.: # highest 5000K
        state_norm[0]=3.
        print("temperature is too high")
    if state_norm[0].item() <-0.357: # lowest 100k
        state_norm[0] = -0.357
        print("temperature is too low")
    if state_norm[1].item()>17.5:# highest 0.4
        state_norm[1]=17.5
        print("mole fraction is too high") # lowest 0.01
    if state_norm[1].item() < -2:
        state_norm[1] = -2
        print("mole fraction is too low")
    state_est = data_norm_store.state_unnorm(state_norm)  # state estimation: normed-->un-normed
    rebuilt_observation = forward_physical_model.generate_sample(state_est)  # state estimation-->rebuilt observation
    error_group,error_sum = Error_function(data_norm_store=data_norm_store,
                           reconstructed_observation=rebuilt_observation,
                           estimated_state=state_norm)
    return rebuilt_observation,state_est,error_group, error_sum




