"""
Fit step for gross alignment and scale.
"""

from opencmiss.zinc.field import Field
from opencmiss.zinc.optimisation import Optimisation
from opencmiss.zinc.result import RESULT_OK, RESULT_WARNING_PART_DONE
from scaffoldfitter.utils.zinc_utils import assignFieldParameters, createFieldEulerAnglesRotationMatrix, getNodeLabelCentres
from scaffoldfitter.scaffit import Scaffit, FitStep

class FitStepAlign(FitStep):

    def __init__(self, fitter : Scaffit):
        super(FitStepAlign, self).__init__(fitter)
        self._rotation = [ 0.0, 0.0, 0.0 ]
        self._scale = 1.0
        self._translation = [ 0.0, 0.0, 0.0 ]
        self._alignLandmarks = False

    @classmethod
    def getTypeId(cls):
        return "FitStepAlign"

    def isAlignLandmarks(self):
        return self._alignLandmarks

    def setAlignLandmarks(self, alignLandmarks):
        """
        :param alignLandmarks: True to automatically align to landmarks, otherwise False.
        """
        self._alignLandmarks = alignLandmarks

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
        fieldmodule = self._fitter._fieldmodule
        fieldmodule.beginChange()
        if self._alignLandmarks:
            errorString = self._doAlignLandmarks()
            if errorString:
                return errorString
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
        landmarkGroup = self._fitter.getLandmarkGroup()
        if landmarkGroup.isValid():
            landmarkDataGroup, landmarkDataCoordinates = self._fitter.getLandmarkDataFields()[0:2]
            landmarkDataCoordinatesTransformed = fieldmodule.createFieldMatrixMultiply(3, rotationMatrix,
                fieldmodule.createFieldAdd(landmarkDataCoordinates, translation))
            fieldassignment = landmarkDataCoordinates.createFieldassignment(landmarkDataCoordinatesTransformed)
            fieldassignment.setNodeset(landmarkDataGroup)
            result = fieldassignment.assign()
            if result not in [ RESULT_OK, RESULT_WARNING_PART_DONE ]:
                return "Align Step: Failed to rigid body transform landmarks"
        # scale model
        scale = fieldmodule.createFieldConstant(self._scale)
        modelCoordinatesScaled = fieldmodule.createFieldMultiply(modelCoordinates, scale)
        fieldassignment = self._fitter._modelCoordinatesField.createFieldassignment(modelCoordinatesScaled)
        result = fieldassignment.assign()
        if result not in [ RESULT_OK, RESULT_WARNING_PART_DONE ]:
            return "Align Step: Failed to scale model"
        self._fitter.updateModelReferenceCoordinates()
        fieldmodule.endChange()
        return None

    def _doAlignLandmarks(self):
        """
        :return: None on success otherwise error string.
        """
        fieldmodule = self._fitter._fieldmodule
        landmarkGroup = self._fitter.getLandmarkGroup()
        if not landmarkGroup.isValid():
            return "Align Step: No landmark group to align with"
        landmarkPrefix = landmarkGroup.getName()
        modelCoordinates = self._fitter.getModelCoordinatesField()
        componentsCount = modelCoordinates.getNumberOfComponents()

        nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        landmarkModelGroup = landmarkGroup.getFieldNodeGroup(nodes).getNodesetGroup()
        if not landmarkModelGroup.isValid():
            return "Align Step: No landmark model group"
        landmarkModelLocation = fieldmodule.findFieldByName(landmarkPrefix + "_location")
        landmarkModelCoordinates = fieldmodule.createFieldEmbedded(modelCoordinates, landmarkModelLocation)
        landmarkModelLabel = fieldmodule.findFieldByName(landmarkPrefix + "_label")
        if not (landmarkModelCoordinates.isValid() and landmarkModelLabel.isValid()):
            return "Align Step: No landmark coordinates or label fields"
        modelLandmarks = getNodeLabelCentres(landmarkModelGroup, landmarkModelCoordinates, landmarkModelLabel)

        landmarkDataGroup, landmarkDataCoordinates, landmarkDataLabel = self._fitter.getLandmarkDataFields()
        if not landmarkDataGroup.isValid():
            return "Align Step: No landmark data group"
        if not (landmarkDataCoordinates.isValid() and landmarkDataLabel.isValid()):
            return "Align Step: No landmark data coordinates or label fields"
        dataLandmarks = getNodeLabelCentres(landmarkDataGroup, landmarkDataCoordinates, landmarkDataLabel)

        # match model and data landmarks, warn of missing landmarks
        landmarksMap = {}
        for label, modelx in modelLandmarks.items():
            datax = dataLandmarks.get(label)
            if datax:
                landmarksMap[label] = ( modelx, datax )
                print('Align Step: Found landmark ' + label + ' in model and data')
        for label in modelLandmarks:
            if not landmarksMap.get(label):
                print('Warning: Align Step: Model landmark ' + label + ' not found in data')
        for label in dataLandmarks:
            if not landmarksMap.get(label):
                print('Warning: Align Step: Data landmark ' + label + ' not found in model')

        return self._optimiseAlignment(landmarksMap)

    def _optimiseAlignment(self, landmarksMap):
        """
        Calculate transformation from modelCoordinates to dataLandmarks
        over the landmarks, by translating and rotating data, and scaling model.
        :param landmarksMap: dict label -> (modelCoordinates, dataCoordinates)
        :return: None on success otherwise errorString. On success,
        sets transformation parameters in object.
        """
        if len(landmarksMap) < 3:
            return "Align Step: Only " + str(len(landmarksMap)) + " landmarks - need at least 3"

        region = self._fitter._context.createRegion()
        fieldmodule = region.getFieldmodule()
        fieldmodule.beginChange()
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
        landmarkDiff = fieldmodule.createFieldSubtract(dataCoordinatesTransformed, modelCoordinatesScaled)
        objective = fieldmodule.createFieldNodesetSumSquares(landmarkDiff, nodes)

        nodetemplate = nodes.createNodetemplate()
        nodetemplate.defineField(modelCoordinates)
        nodetemplate.defineField(dataCoordinates)
        fieldcache = fieldmodule.createFieldcache()
        result = RESULT_OK
        for label, positions in landmarksMap.items():
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
        fieldmodule.endChange()
        if result != RESULT_OK:
            return "Align Step: Align landmarks optimisation set up failed."

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
            return "Align Step. Align landmarks optimisation failed"

        fieldcache = fieldmodule.createFieldcache()
        result1, self._rotation = rotation.evaluateReal(fieldcache, 3)
        result2, self._scale = scale.evaluateReal(fieldcache, 1)
        result3, self._translation = translation.evaluateReal(fieldcache, 3)
        if (result1 != RESULT_OK) or (result2 != RESULT_OK) or (result3 != RESULT_OK):
            return "Align Step. Align landmarks failed to evaluate transformation"
        return None
