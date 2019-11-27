"""
Fit step for gross alignment and scale.
"""

from opencmiss.zinc.field import Field, FieldFindMeshLocation
from opencmiss.zinc.optimisation import Optimisation
from opencmiss.zinc.result import RESULT_OK, RESULT_WARNING_PART_DONE
from scaffoldfitter.utils.zinc_utils import assignFieldParameters, createDisplacementGradientFields, ZincCacheChanges
from scaffoldfitter.fitter import Fitter, FitterStep

class FitterStepFit(FitterStep):

    def __init__(self, fitter : Fitter):
        super(FitterStepFit, self).__init__(fitter)
        self._markerWeight = 1.0
        self._strainPenaltyWeight = 0.0
        self._curvaturePenaltyWeight = 0.0
        self._edgeDiscontinuityPenaltyWeight = 0.0
        self._numberOfIterations = 1
        self._updateReferenceState = False

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
        self._calculateDataProjections()  # Must do first so objectives can be defined
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
            self._calculateDataProjections()
        if self.getDiagnosticLevel() > 0:
            print("--------")

        for d in range(2):
            if dataProjectionObjective[d]:
                result, objective = dataProjectionObjective[d].evaluateReal(fieldcache, dataProjectionObjectiveComponentsCount[d])
                if self.getDiagnosticLevel() > 0:
                    print("END " + str(d + 1) + "-D data projection objective", objective)

        if self._updateReferenceState:
            self._fitter.updateModelReferenceCoordinates()

        return None

    def _calculateDataProjections(self):
        """
        Find projections of datapoints' coordinates onto model coordinates,
        by groups i.e. from datapoints group onto matching 2-D or 1-D mesh group.
        Calculate and store projection direction unit vector.
        Warn about unprojected datapoints.
        """
        fieldmodule = self._fitter._region.getFieldmodule()
        with ZincCacheChanges(fieldmodule):
            dataCoordinates = self._fitter.getDataCoordinatesField()
            modelCoordinates = self._fitter.getModelCoordinatesField()
            dataProjectionDirection = self._fitter.getDataProjectionDirectionField()
            findMeshLocation = None
            #lastDimension = 0
            datapoints = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
            fieldcache = fieldmodule.createFieldcache()
            self._fitter.clearDataProjectionNodesetGroups()
            fielditer = fieldmodule.createFielditerator()
            field = fielditer.next()
            while field.isValid():
                group = field.castGroup()
                field = fielditer.next()
                if group.isValid():
                    groupName = group.getName()
                    dataNodesetGroup = group.getFieldNodeGroup(datapoints).getNodesetGroup()
                    if not dataNodesetGroup.isValid():
                        continue
                    for dimension in range(2, 0, -1):
                        meshGroup = group.getFieldElementGroup(self._fitter.getMesh(dimension)).getMeshGroup()
                        if meshGroup.isValid() and (meshGroup.getSize() > 0):
                            break
                    else:
                        if self.getDiagnosticLevel() > 0:
                            print("Fit Geometry:  Warning: Cannot project data for group " + groupName + " as no matching mesh group")
                        continue
                    meshLocation = self._fitter.getDataProjectionMeshLocationField(dimension)
                    dataProjectionNodesetGroup = self._fitter.getDataProjectionNodesetGroup(dimension)
                    nodeIter = dataNodesetGroup.createNodeiterator()
                    node = nodeIter.next()
                    fieldcache.setNode(node)
                    if not dataCoordinates.isDefinedAtLocation(fieldcache):
                        if self.getDiagnosticLevel() > 0:
                            print("Fit Geometry:  Warning: Cannot project data for group " + groupName + " as field " + dataCoordinates.getName() + " is not defined on data")
                        continue
                    if not meshLocation.isDefinedAtLocation(fieldcache):
                        # define meshLocation and on data Group:
                        nodetemplate = datapoints.createNodetemplate()
                        nodetemplate.defineField(meshLocation)
                        nodetemplate.defineField(dataProjectionDirection)
                        while node.isValid():
                            node.merge(nodetemplate)
                            node = nodeIter.next()
                        del nodetemplate
                        # restart iteration
                        nodeIter = dataNodesetGroup.createNodeiterator()
                        node = nodeIter.next()
                    findMeshLocation = fieldmodule.createFieldFindMeshLocation(dataCoordinates, modelCoordinates, meshGroup)
                    findMeshLocation.setSearchMode(FieldFindMeshLocation.SEARCH_MODE_NEAREST)
                    while node.isValid():
                        fieldcache.setNode(node)
                        element, xi = findMeshLocation.evaluateMeshLocation(fieldcache, dimension)
                        if not element.isValid():
                            print("Fit Geometry:  Error finding data projection nearest mesh location for group " + groupName + ". Aborting group.")
                            break
                        result = meshLocation.assignMeshLocation(fieldcache, element, xi)
                        assert result == RESULT_OK, "Fit Geometry:  Failed to assign data projection mesh location for group " + groupName
                        dataProjectionNodesetGroup.addNode(node)
                        node = nodeIter.next()

            # Store data projection directions
            for dimension in range(1, 3):
                nodesetGroup = self._fitter.getDataProjectionNodesetGroup(dimension)
                if nodesetGroup.getSize() > 0:
                    fieldassignment = dataProjectionDirection.createFieldassignment(
                        fieldmodule.createFieldNormalise(fieldmodule.createFieldSubtract(
                            fieldmodule.createFieldEmbedded(modelCoordinates, self._fitter.getDataProjectionMeshLocationField(dimension)),
                            dataCoordinates)))
                    fieldassignment.setNodeset(nodesetGroup)
                    result = fieldassignment.assign()
                    assert result in [ RESULT_OK, RESULT_WARNING_PART_DONE ], \
                        "Fit Geometry:  Failed to assign data projection directions for dimension " + str(dimension)

            if self.getDiagnosticLevel() > 0:
                # Warn about unprojected points
                unprojectedDatapoints = fieldmodule.createFieldNodeGroup(datapoints).getNodesetGroup()
                unprojectedDatapoints.addNodesConditional(fieldmodule.createFieldIsDefined(dataCoordinates))
                for d in range(2):
                    unprojectedDatapoints.removeNodesConditional(self._fitter.getDataProjectionNodeGroupField(d + 1))
                unprojectedCount = unprojectedDatapoints.getSize()
                if unprojectedCount > 0:
                    print("Warning: " + str(unprojected) + " data points with data coordinates have not been projected")
                del unprojectedDatapoints

            # remove temporary objects before clean up ZincCacheChanges
            del findMeshLocation
            del fieldcache
            del fielditer


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
            dataProjectionCoordinates = fieldmodule.createFieldEmbedded(self._fitter.getModelCoordinatesField(), self._fitter.getDataProjectionMeshLocationField(dimension))
            dataProjectionDelta = dataProjectionCoordinates - self._fitter.getDataCoordinatesField()
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
            markerDataCoordinates = self._fitter.getMarkerDataFields()[1]
            markerDataLocationCoordinates = fieldmodule.createFieldEmbedded(self._fitter.getModelCoordinatesField(), self._fitter.getMarkerDataLocationField())
            markerDataDelta = markerDataLocationCoordinates - markerDataCoordinates
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
