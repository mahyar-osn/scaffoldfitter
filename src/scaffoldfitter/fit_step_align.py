"""
Fit step for gross alignment and scale.
"""

from opencmiss.zinc.field import Field
from opencmiss.zinc.optimisation import Optimisation
from opencmiss.zinc.result import RESULT_OK, RESULT_WARNING_PART_DONE
from scaffoldfitter.utils.zinc_utils import assignFieldParameters, createFieldEulerAnglesRotationMatrix, getNodeLabelCentres, ZincCacheChanges
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
        :return: None on success otherwise error string.
        """
        modelCoordinates = self._fitter.getModelCoordinatesField()
        dataCoordinates = self._fitter.getDataCoordinatesField()
        if (not modelCoordinates) or (not dataCoordinates):
            return "Align Step failed: data and/or model coordinate field not specified"
        if self._alignMarkers:
            errorString = self._doAlignMarkers()
            if errorString:
                return errorString
        fieldmodule = self._fitter._fieldmodule
        with ZincCacheChanges(fieldmodule):
            # translate and rotate data
            translation = fieldmodule.createFieldConstant(self._translation)
            rotation = fieldmodule.createFieldConstant(self._rotation)
            rotationMatrix = createFieldEulerAnglesRotationMatrix(fieldmodule, rotation)
            dataCoordinatesTransformed = fieldmodule.createFieldMatrixMultiply(3, rotationMatrix,
                fieldmodule.createFieldAdd(dataCoordinates, translation))
            fieldassignment = self._fitter._dataCoordinatesField.createFieldassignment(dataCoordinatesTransformed)
            fieldassignment.setNodeset(fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS))
            result = fieldassignment.assign()
            if result not in [ RESULT_OK, RESULT_WARNING_PART_DONE ]:
                return "Align Step: Failed to rigid body transform data"
            markerGroup = self._fitter.getMarkerGroup()
            if markerGroup.isValid():
                markerDataGroup, markerDataCoordinates = self._fitter.getMarkerDataFields()[0:2]
                markerDataCoordinatesTransformed = fieldmodule.createFieldMatrixMultiply(3, rotationMatrix,
                    fieldmodule.createFieldAdd(markerDataCoordinates, translation))
                fieldassignment = markerDataCoordinates.createFieldassignment(markerDataCoordinatesTransformed)
                fieldassignment.setNodeset(markerDataGroup)
                result = fieldassignment.assign()
                if result not in [ RESULT_OK, RESULT_WARNING_PART_DONE ]:
                    return "Align Step: Failed to rigid body transform markers"
            # scale model
            scale = fieldmodule.createFieldConstant(self._scale)
            modelCoordinatesScaled = fieldmodule.createFieldMultiply(modelCoordinates, scale)
            fieldassignment = self._fitter._modelCoordinatesField.createFieldassignment(modelCoordinatesScaled)
            result = fieldassignment.assign()
            if result not in [ RESULT_OK, RESULT_WARNING_PART_DONE ]:
                return "Align Step: Failed to scale model"
            self._fitter.updateModelReferenceCoordinates()
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

        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        markerModelGroup = markerGroup.getFieldNodeGroup(nodes).getNodesetGroup()
        if not markerModelGroup.isValid():
            return "Align Step: No marker model group"
        markerModelLocation = fieldmodule.findFieldByName(markerPrefix + "_location")
        markerModelCoordinates = fieldmodule.createFieldEmbedded(modelCoordinates, markerModelLocation)
        markerModelLabel = fieldmodule.findFieldByName(markerPrefix + "_label")
        if not (markerModelCoordinates.isValid() and markerModelLabel.isValid()):
            return "Align Step: No marker coordinates or label fields"
        modelMarkers = getNodeLabelCentres(markerModelGroup, markerModelCoordinates, markerModelLabel)

        markerDataGroup, markerDataCoordinates, markerDataLabel = self._fitter.getMarkerDataFields()
        if not markerDataGroup.isValid():
            return "Align Step: No marker data group"
        if not (markerDataCoordinates.isValid() and markerDataLabel.isValid()):
            return "Align Step: No marker data coordinates or label fields"
        dataMarkers = getNodeLabelCentres(markerDataGroup, markerDataCoordinates, markerDataLabel)

        # match model and data markers, warn of missing markers
        markerMap = {}
        for label, modelx in modelMarkers.items():
            datax = dataMarkers.get(label)
            if datax:
                markerMap[label] = ( modelx, datax )
                print('Align Step: Found marker ' + label + ' in model and data')
        for label in modelMarkers:
            if not markerMap.get(label):
                print('Warning: Align Step: Model marker ' + label + ' not found in data')
        for label in dataMarkers:
            if not markerMap.get(label):
                print('Warning: Align Step: Data marker ' + label + ' not found in model')

        return self._optimiseAlignment(markerMap)

    def _optimiseAlignment(self, markerMap):
        """
        Calculate transformation from modelCoordinates to dataMarkers
        over the markers, by translating and rotating data, and scaling model.
        :param markerMap: dict label -> (modelCoordinates, dataCoordinates)
        :return: None on success otherwise errorString. On success,
        sets transformation parameters in object.
        """
        if len(markerMap) < 3:
            return "Align Step: Only " + str(len(markerMap)) + " markers - need at least 3"

        region = self._fitter._context.createRegion()
        fieldmodule = region.getFieldmodule()
        with ZincCacheChanges(fieldmodule):
            modelCoordinates = fieldmodule.createFieldFiniteElement(3)
            dataCoordinates = fieldmodule.createFieldFiniteElement(3)
            nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
            # translate and rotate data
            translation = fieldmodule.createFieldConstant([ 0.0, 0.0, 0.0 ])
            rotation = fieldmodule.createFieldConstant([ 0.0, 0.0, 0.0 ])
            rotationMatrix = createFieldEulerAnglesRotationMatrix(fieldmodule, rotation)
            dataCoordinatesTransformed = fieldmodule.createFieldMatrixMultiply(3, rotationMatrix,
                fieldmodule.createFieldAdd(dataCoordinates, translation))
            # scale model
            scale = fieldmodule.createFieldConstant([ 1.0 ])
            modelCoordinatesScaled = fieldmodule.createFieldMultiply(modelCoordinates, scale)
            # create objective = distance from dataCoordinatesTransformed to modelCoordinatesScaled
            markerDiff = fieldmodule.createFieldSubtract(dataCoordinatesTransformed, modelCoordinatesScaled)
            objective = fieldmodule.createFieldNodesetSumSquares(markerDiff, nodes)

            nodetemplate = nodes.createNodetemplate()
            nodetemplate.defineField(modelCoordinates)
            nodetemplate.defineField(dataCoordinates)
            fieldcache = fieldmodule.createFieldcache()
            result = RESULT_OK
            for label, positions in markerMap.items():
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
        if result != RESULT_OK:
            return "Align Step: Align markers optimisation set up failed."

        # future: prefit to avoid gimbal lock

        optimisation = fieldmodule.createOptimisation()
        optimisation.setMethod(Optimisation.METHOD_LEAST_SQUARES_QUASI_NEWTON)
        optimisation.addObjectiveField(objective)
        optimisation.addIndependentField(translation)
        optimisation.addIndependentField(scale)
        optimisation.addIndependentField(rotation)

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
