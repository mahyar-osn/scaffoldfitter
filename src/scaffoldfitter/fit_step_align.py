"""
Fit step for gross alignment and scale.
"""

from opencmiss.zinc.field import Field
from opencmiss.zinc.optimisation import Optimisation
from opencmiss.zinc.result import RESULT_OK, RESULT_WARNING_PART_DONE
from scaffoldfitter.utils.zinc_utils import assignFieldParameters, createFieldEulerAnglesRotationMatrix, \
    createTransformationFields, getNodeNameCentres, ZincCacheChanges
from scaffoldfitter.scaffit import Scaffit, FitStep

class FitStepAlign(FitStep):

    def __init__(self, fitter : Scaffit):
        super(FitStepAlign, self).__init__(fitter)
        self._rotation = [ 0.0, 0.0, 0.0 ]
        self._scale = 1.0
        self._translation = [ 0.0, 0.0, 0.0 ]
        self._alignMarkers = False

    @classmethod
    def getTypeId(cls):
        return "FitStepAlign"

    def isAlignMarkers(self):
        return self._alignMarkers

    def setAlignMarkers(self, alignMarkers):
        """
        :param alignMarkers: True to automatically align to markers, otherwise False.
        """
        self._alignMarkers = alignMarkers

    def getRotation(self):
        return self._rotation

    def setRotation(self, rotation):
        """
        :param rotation: List of 3 euler angles in radians, order applied:
        0 = azimuth (about z)
        1 = elevation (about rotated y)
        2 = roll (about rotated x)
        """
        self._rotation = rotation

    def getScale(self):
        return self._scale

    def setScale(self, scale):
        """
        :param scale: Real scale.
        """
        self._scale = scale

    def getTranslation(self):
        return self._translation

    def setTranslation(self, translation):
        """
        :param translation: [ x, y, z ].
        """
        self._translation = translation

    def run(self):
        """
        Perform align and scale.
        """
        modelCoordinates = self._fitter.getModelCoordinatesField()
        if not modelCoordinates:
            return "Align Step failed: model coordinates field not specified"
        if self._alignMarkers:
            errorString = self._doAlignMarkers()
            if errorString:
                return errorString
        fieldmodule = self._fitter._fieldmodule
        with ZincCacheChanges(fieldmodule):
            # rotate, scale and translate model
            modelCoordinatesTransformed = createTransformationFields(
                modelCoordinates, self._rotation, self._scale, self._translation)[0]
            fieldassignment = self._fitter._modelCoordinatesField.createFieldassignment(modelCoordinatesTransformed)
            result = fieldassignment.assign()
            if result not in [ RESULT_OK, RESULT_WARNING_PART_DONE ]:
                return "Align Step: Failed to transform model"
            self._fitter.updateModelReferenceCoordinates()
            del fieldassignment
            del modelCoordinatesTransformed
        return None

    def _doAlignMarkers(self):
        """
        :return: None on success otherwise error string.
        """
        fieldmodule = self._fitter._fieldmodule
        markerGroup = self._fitter.getMarkerGroup()
        if not markerGroup.isValid():
            return "Align Step: No marker group to align with"
        markerPrefix = markerGroup.getName()
        modelCoordinates = self._fitter.getModelCoordinatesField()
        componentsCount = modelCoordinates.getNumberOfComponents()

        markerNodeGroup, markerLocation, markerName = self._fitter.getMarkerModelFields()
        if not markerNodeGroup.isValid():
            return "Align Step: No marker model group"
        markerModelCoordinates = fieldmodule.createFieldEmbedded(modelCoordinates, markerLocation)
        if not (markerModelCoordinates.isValid() and markerName.isValid()):
            return "Align Step: No marker coordinates or name fields"
        modelMarkers = getNodeNameCentres(markerNodeGroup, markerModelCoordinates, markerName)

        markerDataGroup, markerDataCoordinates, markerDataName = self._fitter.getMarkerDataFields()
        if not markerDataGroup.isValid():
            return "Align Step: No marker data group"
        if not (markerDataCoordinates.isValid() and markerDataName.isValid()):
            return "Align Step: No marker data coordinates or name fields"
        dataMarkers = getNodeNameCentres(markerDataGroup, markerDataCoordinates, markerDataName)

        # match model and data markers, warn of missing markers
        markerMap = {}
        for name, modelx in modelMarkers.items():
            datax = dataMarkers.get(name)
            if datax:
                markerMap[name] = ( modelx, datax )
                print('Align Step: Found marker ' + name + ' in model and data')
        for name in modelMarkers:
            if not markerMap.get(name):
                print('Warning: Align Step: Model marker ' + name + ' not found in data')
        for name in dataMarkers:
            if not markerMap.get(name):
                print('Warning: Align Step: Data marker ' + name + ' not found in model')

        return self._optimiseAlignment(markerMap)

    def _optimiseAlignment(self, markerMap):
        """
        Calculate transformation from modelCoordinates to dataMarkers
        over the markers, by scaling, translating and rotating model.
        :param markerMap: dict name -> (modelCoordinates, dataCoordinates)
        :return: None on success otherwise errorString. On success,
        sets transformation parameters in object.
        """
        if len(markerMap) < 3:
            return "Align Step: Only " + str(len(markerMap)) + " markers - need at least 3"

        region = self._fitter._context.createRegion()
        fieldmodule = region.getFieldmodule()
        result = RESULT_OK
        with ZincCacheChanges(fieldmodule):
            modelCoordinates = fieldmodule.createFieldFiniteElement(3)
            dataCoordinates = fieldmodule.createFieldFiniteElement(3)
            nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
            nodetemplate = nodes.createNodetemplate()
            nodetemplate.defineField(modelCoordinates)
            nodetemplate.defineField(dataCoordinates)
            fieldcache = fieldmodule.createFieldcache()
            for name, positions in markerMap.items():
                modelx = positions[0]
                datax = positions[1]
                node = nodes.createNode(-1, nodetemplate)
                fieldcache.setNode(node)
                result = modelCoordinates.assignReal(fieldcache, positions[0])
                if result != RESULT_OK:
                    break
                result = dataCoordinates.assignReal(fieldcache, positions[1])
                if result != RESULT_OK:
                    break
            del fieldcache
            modelCoordinatesTransformed, rotation, scale, translation = createTransformationFields(modelCoordinates)
            # create objective = sum of squares of vector from modelCoordinatesTransformed to dataCoordinates
            markerDiff = fieldmodule.createFieldSubtract(dataCoordinates, modelCoordinatesTransformed)
            objective = fieldmodule.createFieldNodesetSumSquares(markerDiff, nodes)

        if result != RESULT_OK:
            return "Align Step: Align markers optimisation set up failed."

        # future: prefit to avoid gimbal lock

        optimisation = fieldmodule.createOptimisation()
        optimisation.setMethod(Optimisation.METHOD_LEAST_SQUARES_QUASI_NEWTON)
        optimisation.addObjectiveField(objective)
        optimisation.addIndependentField(rotation)
        optimisation.addIndependentField(scale)
        optimisation.addIndependentField(translation)

        result = optimisation.optimise()
        solutionReport = optimisation.getSolutionReport()
        print('Align result', result)
        print(solutionReport)
        if result != RESULT_OK:
            return "Align Step. Align markers optimisation failed"

        fieldcache = fieldmodule.createFieldcache()
        result1, self._rotation = rotation.evaluateReal(fieldcache, 3)
        result2, self._scale = scale.evaluateReal(fieldcache, 1)
        result3, self._translation = translation.evaluateReal(fieldcache, 3)
        if (result1 != RESULT_OK) or (result2 != RESULT_OK) or (result3 != RESULT_OK):
            return "Align Step. Align markers failed to evaluate transformation"
        return None
