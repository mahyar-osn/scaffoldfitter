"""
Fit step for gross alignment and scale.
"""

from opencmiss.utils.zinc.field import assignFieldParameters, createDisplacementGradientFields
from opencmiss.utils.zinc.general import ZincCacheChanges
from opencmiss.zinc.field import Field, FieldFindMeshLocation
from opencmiss.zinc.optimisation import Optimisation
from opencmiss.zinc.result import RESULT_OK
from scaffoldfitter.fitter import Fitter, FitterStep

class FitterStepFit(FitterStep):

    _jsonTypeId = "_FitterStepFit"

    def __init__(self, fitter : Fitter):
        super(FitterStepFit, self).__init__(fitter)
        self._markerWeight = 1.0
        self._strainPenaltyWeight = 0.0
        self._curvaturePenaltyWeight = 0.0
        self._edgeDiscontinuityPenaltyWeight = 0.0
        self._numberOfIterations = 1
        self._updateReferenceState = False

    @classmethod
    def getJsonTypeId(cls):
        return cls._jsonTypeId

    def decodeSettingsJSONDict(self, dct : dict):
        """
        Decode definition of step from JSON dict.
        """
        assert self._jsonTypeId in dct
        self._markerWeight = dct["markerWeight"]
        self._strainPenaltyWeight = dct["strainPenaltyWeight"]
        self._curvaturePenaltyWeight = dct["curvaturePenaltyWeight"]
        self._edgeDiscontinuityPenaltyWeight = dct["edgeDiscontinuityPenaltyWeight"]
        self._numberOfIterations = dct["numberOfIterations"]
        self._updateReferenceState = dct["updateReferenceState"]

    def encodeSettingsJSONDict(self) -> dict:
        """
        Encode definition of step in dict.
        :return: Settings in a dict ready for passing to json.dump.
        """
        return {
            self._jsonTypeId : True,
            "markerWeight" : self._markerWeight,
            "strainPenaltyWeight" : self._strainPenaltyWeight,
            "curvaturePenaltyWeight" : self._curvaturePenaltyWeight,
            "edgeDiscontinuityPenaltyWeight" : self._edgeDiscontinuityPenaltyWeight,
            "numberOfIterations" : self._numberOfIterations,
            "updateReferenceState" : self._updateReferenceState
            }

    def getMarkerWeight(self):
        return self._markerWeight

    def setMarkerWeight(self, weight):
        assert weight >= 0.0
        self._markerWeight = weight

    def getStrainPenaltyWeight(self):
        return self._strainPenaltyWeight

    def setStrainPenaltyWeight(self, weight):
        assert weight >= 0.0
        self._strainPenaltyWeight = weight

    def getCurvaturePenaltyWeight(self):
        return self._curvaturePenaltyWeight

    def setCurvaturePenaltyWeight(self, weight):
        assert weight >= 0.0
        self._curvaturePenaltyWeight = weight

    def getEdgeDiscontinuityPenaltyWeight(self):
        return self._edgeDiscontinuityPenaltyWeight

    def setEdgeDiscontinuityPenaltyWeight(self, weight):
        assert weight >= 0.0
        self._edgeDiscontinuityPenaltyWeight = weight

    def getNumberOfIterations(self):
        return self._numberOfIterations

    def setNumberOfIterations(self, numberOfIterations):
        assert numberOfIterations >= 0  # 0 iterations performs projections only
        self._numberOfIterations = numberOfIterations

    def isUpdateReferenceState(self):
        return self._updateReferenceState

    def setUpdateReferenceState(self, updateReferenceState):
        self._updateReferenceState = updateReferenceState

    def run(self):
        """
        Fit model geometry parameters to data.
        """
        fieldmodule = self._fitter._region.getFieldmodule()
        optimisation = fieldmodule.createOptimisation()
        optimisation.setMethod(Optimisation.METHOD_LEAST_SQUARES_QUASI_NEWTON)
        optimisation.addIndependentField(self._fitter.getModelCoordinatesField())
        optimisation.setAttributeInteger(Optimisation.ATTRIBUTE_MAXIMUM_ITERATIONS, 5)

        dataProjectionObjective = [ None, None ]
        dataProjectionObjectiveComponentsCount = [ 0, 0 ]
        markerObjectiveField = None
        deformationPenaltyObjective = None
        edgeDiscontinuityPenaltyObjective = None
        for dimension in range(1, 3):
            if self._fitter.getDataProjectionNodesetGroup(dimension).getSize() > 0:
                dataProjectionObjective[dimension - 1] = self.createDataProjectionObjectiveField(dimension)
                dataProjectionObjectiveComponentsCount[dimension - 1] = dataProjectionObjective[dimension - 1].getNumberOfComponents()
                result = optimisation.addObjectiveField(dataProjectionObjective[dimension - 1])
                assert result == RESULT_OK, "Fit Geometry:  Could not add data projection objective field for dimension " + str(dimension)
        if self._fitter.getMarkerGroup() and (self._fitter.getMarkerDataLocationNodesetGroup().getSize() > 0) and (self._markerWeight > 0.0):
            markerObjectiveField = self.createMarkerObjectiveField(self._markerWeight)
            result = optimisation.addObjectiveField(markerObjectiveField)
            assert result == RESULT_OK, "Fit Geometry:  Could not add marker objective field"
        if (self._strainPenaltyWeight > 0.0) or (self._curvaturePenaltyWeight > 0.0):
            deformationPenaltyObjective = self.createDeformationPenaltyObjectiveField()
            result = optimisation.addObjectiveField(deformationPenaltyObjective)
            assert result == RESULT_OK, "Fit Geometry:  Could not add strain/curvature penalty objective field"
        if self._edgeDiscontinuityPenaltyWeight > 0.0:
            edgeDiscontinuityPenaltyObjective = self.createEdgeDiscontinuityPenaltyObjectiveField()
            result = optimisation.addObjectiveField(edgeDiscontinuityPenaltyObjective)
            assert result == RESULT_OK, "Fit Geometry:  Could not add edge discontinuity penalty objective field"

        fieldcache = fieldmodule.createFieldcache()
        for iter in range(self._numberOfIterations):
            if self.getDiagnosticLevel() > 0:
                print("-------- Iteration", iter + 1)
            for d in range(2):
                if dataProjectionObjective[d]:
                    result, objective = dataProjectionObjective[d].evaluateReal(fieldcache, dataProjectionObjectiveComponentsCount[d])
                    if self.getDiagnosticLevel() > 0:
                        print("    " + str(d + 1) + "-D data projection objective", objective)
            result = optimisation.optimise()
            if self.getDiagnosticLevel() > 1:
                solutionReport = optimisation.getSolutionReport()
                print(solutionReport)
            assert result == RESULT_OK, "Fit Geometry:  Optimisation failed with result " + str(result)
            self._fitter.calculateDataProjections()
        if self.getDiagnosticLevel() > 0:
            print("--------")

        for d in range(2):
            if dataProjectionObjective[d]:
                result, objective = dataProjectionObjective[d].evaluateReal(fieldcache, dataProjectionObjectiveComponentsCount[d])
                if self.getDiagnosticLevel() > 0:
                    print("END " + str(d + 1) + "-D data projection objective", objective)

        if self._updateReferenceState:
            self._fitter.updateModelReferenceCoordinates()

        self.setHasRun(True)

    def createDataProjectionObjectiveField(self, dimension):
        """
        Get FieldNodesetSumSquares objective for data projected onto mesh of dimension.
        Minimises length in projection direction, allowing sliding fit.
        Only call if self._fitter.getDataProjectionNodesetGroup().getSize() > 0
        :param dimension: Mesh dimension 1 or 2.
        :return: Zinc FieldNodesetSumSquares.
        """
        fieldmodule = self._fitter.getFieldmodule()
        with ZincCacheChanges(fieldmodule):
            dataProjectionDelta = self._fitter.getDataProjectionDeltaField(dimension)
            #dataProjectionInDirection = fieldmodule.createFieldDotProduct(dataProjectionDelta, self._fitter.getDataProjectionDirectionField())
            #dataProjectionInDirection = fieldmodule.createFieldMagnitude(dataProjectionDelta)
            dataProjectionInDirection = dataProjectionDelta
            dataProjectionObjective = fieldmodule.createFieldNodesetSumSquares(dataProjectionInDirection, self._fitter.getDataProjectionNodesetGroup(dimension))
        return dataProjectionObjective

    def createMarkerObjectiveField(self, weight):
        """
        Only call if self._fitter.getMarkerGroup() and (self._fitter.getMarkerDataLocationNodesetGroup().getSize() > 0) and (self._markerWeight > 0.0)
        For marker datapoints with locations in model, creates a FieldNodesetSumSquares
        of coordinate projections to those locations.
        :return: Zinc FieldNodesetSumSquares.
        """
        fieldmodule = self._fitter.getFieldmodule()
        with ZincCacheChanges(fieldmodule):
            markerDataLocation, markerDataLocationCoordinates, markerDataDelta = self._fitter.getMarkerDataLocationFields()
            markerDataWeightedDelta = markerDataDelta*fieldmodule.createFieldConstant([ weight ]*markerDataDelta.getNumberOfComponents())
            markerDataObjective = fieldmodule.createFieldNodesetSumSquares(markerDataWeightedDelta, self._fitter.getMarkerDataLocationNodesetGroup())
        return markerDataObjective

    def createDeformationPenaltyObjectiveField(self):
        """
        Only call if (self._strainPenaltyWeight > 0.0) or (self._curvaturePenaltyWeight > 0.0)
        :return: Zinc FieldMeshIntegralSquares, or None if not weighted.
        """
        numberOfGaussPoints = 3
        fieldmodule = self._fitter.getFieldmodule()
        mesh = self._fitter.getHighestDimensionMesh()
        with ZincCacheChanges(fieldmodule):
            displacementGradient1, displacementGradient2 = createDisplacementGradientFields(self._fitter.getModelCoordinatesField(), self._fitter.getModelReferenceCoordinatesField(), mesh)
            if self._strainPenaltyWeight > 0.0:
                weightedDisplacementGradient1 = displacementGradient1*fieldmodule.createFieldConstant([ self._strainPenaltyWeight ]*displacementGradient1.getNumberOfComponents())
            else:
                weightedDisplacementGradient1 = None
            if self._curvaturePenaltyWeight > 0.0:
                weightedDisplacementGradient2 = displacementGradient2*fieldmodule.createFieldConstant([ self._curvaturePenaltyWeight ]*displacementGradient2.getNumberOfComponents())
            else:
                weightedDisplacementGradient2 = None

            if weightedDisplacementGradient1:
                if weightedDisplacementGradient2:
                    deformationField = fieldmodule.createFieldConcatenate([ weightedDisplacementGradient1, weightedDisplacementGradient2 ])
                else:
                    deformationField = weightedDisplacementGradient1
            elif weightedDisplacementGradient2:
                deformationField = weightedDisplacementGradient2
            else:
                return None

            deformationPenaltyObjective = fieldmodule.createFieldMeshIntegralSquares(deformationField, self._fitter.getModelReferenceCoordinatesField(), mesh)
            deformationPenaltyObjective.setNumbersOfPoints(numberOfGaussPoints)
        return deformationPenaltyObjective

    def createEdgeDiscontinuityPenaltyObjectiveField(self):
        """
        Only call if self._edgeDiscontinuityPenaltyWeight > 0.0
        :return: Zinc FieldMeshIntegralSquares, or None if not weighted.
        """
        numberOfGaussPoints = 3
        fieldmodule = self._fitter.getFieldmodule()
        lineMesh = fieldmodule.findMeshByDimension(1)
        with ZincCacheChanges(fieldmodule):
            edgeDiscontinuity = fieldmodule.createFieldEdgeDiscontinuity(self._fitter.getModelCoordinatesField())
            weightedEdgeDiscontinuity = edgeDiscontinuity*fieldmodule.createFieldConstant(self._edgeDiscontinuityPenaltyWeight)
            edgeDiscontinuityPenaltyObjective = fieldmodule.createFieldMeshIntegralSquares(weightedEdgeDiscontinuity, self._fitter.getModelReferenceCoordinatesField(), lineMesh)
            edgeDiscontinuityPenaltyObjective.setNumbersOfPoints(numberOfGaussPoints)
        return edgeDiscontinuityPenaltyObjective
