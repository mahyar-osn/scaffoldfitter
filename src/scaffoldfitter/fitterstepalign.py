"""
Fit step for gross alignment and scale.
"""

from opencmiss.utils.zinc.field import assignFieldParameters, createFieldsTransformations
from opencmiss.utils.zinc.finiteelement import getNodeNameCentres
from opencmiss.utils.zinc.general import ChangeManager
from opencmiss.zinc.field import Field
from opencmiss.zinc.optimisation import Optimisation
from opencmiss.zinc.result import RESULT_OK, RESULT_WARNING_PART_DONE
from scaffoldfitter.fitter import Fitter, FitterStep

class FitterStepAlign(FitterStep):

    _jsonTypeId = "_FitterStepAlign"

    def __init__(self, fitter : Fitter):
        super(FitterStepAlign, self).__init__(fitter)
        self._alignMarkers = False
        markerNodeGroup, markerLocation, markerCoordinates, markerName = fitter.getMarkerModelFields()
        markerDataGroup, markerDataCoordinates, markerDataName = fitter.getMarkerDataFields()
        if markerNodeGroup and markerLocation and markerCoordinates and markerName and \
            markerDataGroup and markerDataCoordinates and markerDataName:
            self._alignMarkers = True
        self._rotation = [ 0.0, 0.0, 0.0 ]
        self._scale = 1.0
        self._translation = [ 0.0, 0.0, 0.0 ]

    @classmethod
    def getJsonTypeId(cls):
        return cls._jsonTypeId

    def decodeSettingsJSONDict(self, dct : dict):
        """
        Decode definition of step from JSON dict.
        """
        assert self._jsonTypeId in dct
        self._alignMarkers = dct["alignMarkers"]
        self._rotation = dct["rotation"]
        self._scale = dct["scale"]
        self._translation = dct["translation"]

    def encodeSettingsJSONDict(self) -> dict:
        """
        Encode definition of step in dict.
        :return: Settings in a dict ready for passing to json.dump.
        """
        return {
            self._jsonTypeId : True,
            "alignMarkers" : self._alignMarkers,
            "rotation" : self._rotation,
            "scale" : self._scale,
            "translation" : self._translation
            }

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
        assert len(rotation) == 3, "FitterStepAlign:  Invalid rotation"
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
        assert len(translation) == 3, "FitterStepAlign:  Invalid translation"
        self._translation = translation

    def run(self):
        """
        Perform align and scale.
        """
        modelCoordinates = self._fitter.getModelCoordinatesField()
        assert modelCoordinates, "Align:  Missing model coordinates"
        if self._alignMarkers:
            self._doAlignMarkers()
        fieldmodule = self._fitter._fieldmodule
        with ChangeManager(fieldmodule):
            # rotate, scale and translate model
            modelCoordinatesTransformed = createFieldsTransformations(
                modelCoordinates, self._rotation, self._scale, self._translation)[0]
            fieldassignment = self._fitter._modelCoordinatesField.createFieldassignment(modelCoordinatesTransformed)
            result = fieldassignment.assign()
            assert result in [ RESULT_OK, RESULT_WARNING_PART_DONE ], "Align:  Failed to transform model"
            self._fitter.updateModelReferenceCoordinates()
            del fieldassignment
            del modelCoordinatesTransformed
        self._fitter.calculateDataProjections()
        self.setHasRun(True)

    def _doAlignMarkers(self):
        """
        Prepare and invoke alignment to markers.
        """
        fieldmodule = self._fitter._fieldmodule
        markerGroup = self._fitter.getMarkerGroup()
        assert markerGroup, "Align:  No marker group to align with"
        markerPrefix = markerGroup.getName()
        modelCoordinates = self._fitter.getModelCoordinatesField()
        componentsCount = modelCoordinates.getNumberOfComponents()

        markerNodeGroup, markerLocation, markerCoordinates, markerName = self._fitter.getMarkerModelFields()
        assert markerNodeGroup and markerCoordinates and markerName, "Align:  No marker group, coordinates or name fields"
        modelMarkers = getNodeNameCentres(markerNodeGroup, markerCoordinates, markerName)

        markerDataGroup, markerDataCoordinates, markerDataName = self._fitter.getMarkerDataFields()
        assert markerDataGroup and markerDataCoordinates and markerDataName, "Align:  No marker data group, coordinates or name fields"
        dataMarkers = getNodeNameCentres(markerDataGroup, markerDataCoordinates, markerDataName)

        # match model and data markers, warn of missing markers
        markerMap = {}
        for name, modelx in modelMarkers.items():
            datax = dataMarkers.get(name)
            if datax:
                markerMap[name] = ( modelx, datax )
        if self.getDiagnosticLevel() > 0:
            for name in modelMarkers:
                datax = dataMarkers.get(name)
                if datax:
                    print("Align:  Found marker " + name + " in model and data")
            for name in modelMarkers:
                if not markerMap.get(name):
                    print("Align:  Model marker " + name + " not found in data")
            for name in dataMarkers:
                if not markerMap.get(name):
                    print("Align:  Data marker " + name + " not found in model")

        self._optimiseAlignment(markerMap)

    def _optimiseAlignment(self, markerMap):
        """
        Calculate transformation from modelCoordinates to dataMarkers
        over the markers, by scaling, translating and rotating model.
        On success, sets transformation parameters in object.
        :param markerMap: dict name -> (modelCoordinates, dataCoordinates)
        """
        assert len(markerMap) >= 3, "Align:  Only " + str(len(markerMap)) + " markers - need at least 3"
        region = self._fitter._context.createRegion()
        fieldmodule = region.getFieldmodule()
        with ChangeManager(fieldmodule):
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
                result1 = modelCoordinates.assignReal(fieldcache, positions[0])
                result2 = dataCoordinates.assignReal(fieldcache, positions[1])
                assert (result1 == RESULT_OK) and (result2 == RESULT_OK), "Align:  Failed to set up data for alignment to markers optimisation"
            del fieldcache
            modelCoordinatesTransformed, rotation, scale, translation = createFieldsTransformations(modelCoordinates)
            # create objective = sum of squares of vector from modelCoordinatesTransformed to dataCoordinates
            markerDiff = fieldmodule.createFieldSubtract(dataCoordinates, modelCoordinatesTransformed)
            objective = fieldmodule.createFieldNodesetSumSquares(markerDiff, nodes)
            assert objective.isValid(), "Align:  Failed to set up objective function for alignment to markers optimisation"

        # future: pre-fit to avoid gimbal lock

        optimisation = fieldmodule.createOptimisation()
        optimisation.setMethod(Optimisation.METHOD_LEAST_SQUARES_QUASI_NEWTON)
        optimisation.addObjectiveField(objective)
        optimisation.addIndependentField(rotation)
        optimisation.addIndependentField(scale)
        optimisation.addIndependentField(translation)

        result = optimisation.optimise()
        if self.getDiagnosticLevel() > 1:
            solutionReport = optimisation.getSolutionReport()
            print(solutionReport)
        assert result == RESULT_OK, "Align:  Alignment to markers optimisation failed"

        fieldcache = fieldmodule.createFieldcache()
        result1, self._rotation = rotation.evaluateReal(fieldcache, 3)
        result2, self._scale = scale.evaluateReal(fieldcache, 1)
        result3, self._translation = translation.evaluateReal(fieldcache, 3)
        assert (result1 == RESULT_OK) and (result2 == RESULT_OK) and (result3 == RESULT_OK), "Align:  Failed to evaluate transformation for alignment to markers"
