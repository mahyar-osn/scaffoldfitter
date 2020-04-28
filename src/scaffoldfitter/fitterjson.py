"""
Factory function for constructing FitterStep types from JSON Dict.
"""
from scaffoldfitter.fitter import Fitter
from scaffoldfitter.fitterstepalign import FitterStepAlign
from scaffoldfitter.fitterstepfit import FitterStepFit


def decodeJSONFitterSteps(fitter : Fitter, dct):
    """
    Function for passing as decoder to Fitter.decodeSettingsJSON().
    Constructs scaffold objects from their JSON object encoding.
    Used as object_hook argument to json.loads as a lambda function
    to pass fitter:
    lambda dct: decodeJSONFitterSteps(fitter, dct)
    :param fitter: Owning Fitter object for new FitterSteps.
    """
    for FitterStepType in [ FitterStepAlign, FitterStepFit ]:
        if FitterStepType.getJsonTypeId() in dct:
            fitterStep = FitterStepType(fitter)
            fitterStep.decodeSettingsJSONDict(dct)
            return fitterStep
    return dct
