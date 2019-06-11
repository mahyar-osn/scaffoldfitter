from opencmiss.zinc.field import Field, FieldFindMeshLocation
from opencmiss.zinc.status import OK as ZINC_OK
from opencmiss.zinc.glyph import Glyph
from opencmiss.zinc.element import Element
from opencmiss.zinc.graphics import Graphics
from opencmiss.zinc.scenecoordinatesystem import SCENECOORDINATESYSTEM_NORMALISED_WINDOW_FIT_LEFT
from opencmiss.zinc.optimisation import Optimisation

from .utils import maths
from .utils import zincutils
from .utils import field_utils
from .optimization.alignment_fitting import fitRigidScale


class Fitter(object):

    def __init__(self, region, context):
        self._region = region
        self._logger = context.getLogger()
        self._modelReferenceCoordinateField = None
        self._modelCoordinateField = None
        self._alignSettings = None
        self._modelTransformedCoordinateField = None
        self._modelRotationScaleField = None
        self._modelOffsetField = None
        self._dataCoordinateField = None
        self._activeDataPointGroupField = None
        self._mesh = None
        self._modelCentre = None
        self._isStateAlign = True
        self._exteriorSurfaceField = None
        self._exteriorFaceGroup = None
        self._exteriorFaceMeshGroup = None
        self._findMeshLocationField = None
        self._storedMeshLocationField = None
        self._dataProjectionCoordinateField = None
        self._dataProjectionDistanceCoordinateField = None
        self._dataProjectionErrorField = None
        self._dataProjectionMeanErrorField = None
        self._dataProjectionMaximumErrorField = None

        self._resetFitSettings()

    def resetAlignSettings(self):
        self._alignSettings = dict(euler_angles=[0.0, 0.0, 0.0], scale=[1.0]*3, offset=[0.0, 0.0, 0.0], mirror=False)

    def getInitialScale(self):
        dataMinimums, dataMaximums = self._getDataRange()
        dataRange = maths.sub(dataMaximums, dataMinimums)
        modelMinimums, modelMaximums = self._getModelRange()
        modelRange = maths.sub(modelMaximums, modelMinimums)
        tmp = [modelRange[0], modelRange[2], modelRange[1]]
        dataModelDifference = maths.eldiv(dataRange, tmp)
        meanScale = sum(dataModelDifference) / len(dataModelDifference)
        self.setAlignScale(dataModelDifference)
        self.setStatePostAlign()
        self.resetAlignSettings()

    def autoCentreModelOnData(self):
        dataMinimums, dataMaximums = self._getDataRange()
        dataCentre = maths.mult(maths.add(dataMinimums, dataMaximums), 0.5)
        modelMinimums, modelMaximums = self._getModelRange()
        modelCentre = maths.mult(maths.add(modelMinimums, modelMaximums), 0.5)
        self.setAlignOffset(maths.sub(dataCentre, modelCentre))
        self.setStatePostAlign()
        return self._modelCoordinateField

    def initializeRigidAlignment(self):
        scaffoldNodeValus = self._getScaffoldNodeParameters()
        pointCloudValues = self._getPointCloudParameters()
        t0pt, fitted_data, (initial_rms, final_rms), T = fitRigidScale(scaffoldNodeValus, pointCloudValues,
                                                                       xtol=1e-10, maxfev=0, sample=None,
                                                                       scaleThreshold=None, outputErrors=True)

        print("Initial RMS = ", initial_rms)
        print("Final RMS = ", final_rms)

        euler_radians = maths.rotationMatrix3ToEuler(T[:3, :3].tolist())
        euler_radians_correct = [euler_radians[0], euler_radians[1], euler_radians[2]]
        self.setAlignEulerAngles(euler_radians_correct)
        self.setAlignScale(1.)
        self.setStatePostAlign()
        self._isStateAlign = False
        return self._modelCoordinateField

    def swapAxes(self, axes=None):
        zincutils.swap_axes(self._modelCoordinateField, axes=axes)

    def isAlignMirror(self):
        return self._alignSettings['mirror']

    def getAlignEulerAngles(self):
        return self._alignSettings['euler_angles']

    def getAlignScale(self):
        return self._alignSettings['scale']

    def getAlignOffset(self):
        return self._alignSettings['offset']

    def getFitStrainPenalty(self):
        return self._fitSettings['strain_penalty']

    def getFitMaxIterations(self):
        return self._fitSettings['max_iterations']

    def setAlignEulerAngles(self, eulerAngles):
        if len(eulerAngles) == 3:
            self._alignSettings['euler_angles'] = eulerAngles
            self.applyAlignSettings()

    def setAlignMirror(self, mirror):
        self._alignSettings['mirror'] = mirror
        self.applyAlignSettings()

    def setAlignOffset(self, offset):
        if len(offset) == 3:
            self._alignSettings['offset'] = offset
            self.applyAlignSettings()

    def setAlignScale(self, scale):
        self._alignSettings['scale'] = scale
        self.applyAlignSettings()

    # def createFiniteElementField(self, fieldName='coordinates'):
    #     fieldModule = self._region.getFieldmodule()
    #     fieldModule.beginChange()
    #     finiteElementField = fieldModule.createFieldFiniteElement(3)
    #     finiteElementField.setName(fieldName)
    #     finiteElementField.setManaged(True)
    #     finiteElementField.setTypeCoordinate(True)
    #     fieldModule.endChange()
    #     return finiteElementField

    def _getDataRange(self):
        fm = self._region.getFieldmodule()
        datapoints = fm.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        minimums, maximums = self._getNodesetMinimumMaximum(datapoints, self._dataCoordinateField)
        return minimums, maximums

    def getAutoPointSize(self):
        minimums, maximums = self._getDataRange()
        dataSize = maths.magnitude(maths.sub(maximums, minimums))
        return 0.005 * dataSize

    def getRegion(self):
        return self._region

    def coordinatesChanged(self, field):
        self._setRefereceModelCoordinates(field)
        self._setModelCoordinates(field)

    def getMesh(self):
        fm = self._region.getFieldmodule()
        for dimension in range(3, 0, -1):
            mesh = fm.findMeshByDimension(dimension)
            if mesh.getSize() > 0:
                return mesh
        raise ValueError('Model contains no mesh')

    def getModelCoordinateField(self, reference=False):
        mesh = self.getMesh()
        element = mesh.createElementiterator().next()
        if not element.isValid():
            raise ValueError('Model contains no elements')
        fm = self._region.getFieldmodule()
        cache = fm.createFieldcache()
        cache.setElement(element)
        fieldIter = fm.createFielditerator()
        field = fieldIter.next()
        while field.isValid():
            if field.isTypeCoordinate() and (field.getNumberOfComponents() <= 3) and \
                    ((self._modelReferenceCoordinateField is None) or
                     (self._modelReferenceCoordinateField != field) or
                     (self._modelCoordinateField is None) or
                     (self._modelCoordinateField != field)):
                if field.isDefinedAtLocation(cache):
                    if reference:
                        self._setRefereceModelCoordinates(field)
                    else:
                        self._setModelCoordinates(field)
                    return field
            field = fieldIter.next()
        raise ValueError('Could not determine model coordinate field')

    def getDataCoordinateField(self):
        fm = self._region.getFieldmodule()
        datapoints = fm.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        self._activeDataPointGroupField = fm.createFieldNodeGroup(datapoints)
        datapoint = datapoints.createNodeiterator().next()
        if not datapoint.isValid():
            raise ValueError('Point cloud is empty')
        cache = fm.createFieldcache()
        cache.setNode(datapoint)
        fieldIter = fm.createFielditerator()
        field = fieldIter.next()
        while field.isValid():
            if field.isTypeCoordinate() and (field.getNumberOfComponents() <= 3) and \
                    ((self._modelReferenceCoordinateField is None) or (field != self._modelReferenceCoordinateField)):
                if field.isDefinedAtLocation(cache):
                    self._setDataCoordinates(field)
                    return field
            field = fieldIter.next()
        raise ValueError('Could not determine data coordinate field')

    def _getProjectFaceMeshGroup(self):
        fm = self._region.getFieldmodule()
        mesh2d = fm.findMeshByDimension(2)
        self._exteriorFaceGroup = fm.createFieldElementGroup(mesh2d)
        if self._exteriorFaceGroup.isValid() and self._exteriorFaceMeshGroup is None:
            self._exteriorFaceMeshGroup = self._exteriorFaceGroup.getMeshGroup()
            if self._exteriorFaceMeshGroup.isValid():
                isExterior = fm.createFieldIsExterior()
                isOnFaceXi3_1 = fm.createFieldIsOnFace(Element.FACE_TYPE_XI3_1)
                isBoth = fm.createFieldAnd(isExterior, isOnFaceXi3_1)
                if not isBoth.isValid():
                    raise Exception \
                        ("src.scaffoldfitter.fitter.Fitter._getProjectSurfaceGroup: Exterior face was not found!")
                result = self._exteriorFaceMeshGroup.addElementsConditional(isBoth)
                self._exteriorSurfaceField = isBoth
                if result != ZINC_OK:
                    return None
        return None

    def _getStrainField(self, mesh):
        fm = self._region.getFieldmodule()
        dX_dxi1 = fm.createFieldDerivative(self._modelReferenceCoordinateField, 1)
        dX_dxi2 = fm.createFieldDerivative(self._modelReferenceCoordinateField, 2)
        dX_dxi3 = fm.createFieldDerivative(self._modelReferenceCoordinateField, 3)
        FXT = fm.createFieldConcatenate([dX_dxi1, dX_dxi2, dX_dxi3])
        FX = fm.createFieldTranspose(3, FXT)
        CX = fm.createFieldMatrixMultiply(3, FXT, FX)
        dx_dxi1 = fm.createFieldDerivative(self._modelCoordinateField, 1)
        dx_dxi2 = fm.createFieldDerivative(self._modelCoordinateField, 2)
        dx_dxi3 = fm.createFieldDerivative(self._modelCoordinateField, 3)
        FxT = fm.createFieldConcatenate([dx_dxi1, dx_dxi2, dx_dxi3])
        Fx = fm.createFieldTranspose(3, FxT)
        Cx = fm.createFieldMatrixMultiply(3, FxT, Fx)
        strainField = fm.createFieldSubtract(Cx, CX)
        return strainField

    def _resetFitSettings(self):
        self._fitSettings = dict(strain_penalty=.0, edge_discontinuity_penalty=0.0, max_iterations=1)

    def computeProjection(self):
        fm = self._region.getFieldmodule()
        self._getProjectFaceMeshGroup()
        if self._findMeshLocationField is None and self._exteriorFaceMeshGroup is not None:
            self._findMeshLocationField = fm.createFieldFindMeshLocation(self._dataCoordinateField,
                                                                         self._modelCoordinateField,
                                                                         self._exteriorFaceMeshGroup)
            if self._findMeshLocationField.isValid():
                self._findMeshLocationField.setSearchMode(FieldFindMeshLocation.SEARCH_MODE_NEAREST)
            else:
                self._findMeshLocationField = None
                raise ValueError('src.scaffoldfitter.fitter.Fitter.computeProjection:'
                                 'Failed to create find mesh location field.'
                                 'Ensure mesh or data points are correctly defined and initialised.')

        if self._storedMeshLocationField is None:
            self._storedMeshLocationField = fm.createFieldStoredMeshLocation(self._exteriorFaceMeshGroup)
            if not self._storedMeshLocationField.isValid():
                self._storedMeshLocationField = None
                raise ValueError('src.scaffoldfitter.fitter.Fitter.computeProjection:'
                                 'Failed to create stored mesh location field.'
                                 'Ensure the mesh is correctly defined and initialised')

        datapoints = fm.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        tmpTrue = fm.createFieldConstant([1])
        activeDatapointsGroup = self._activeDataPointGroupField.getNodesetGroup()
        activeDatapointsGroup.addNodesConditional(tmpTrue)

        dimension = self._exteriorFaceMeshGroup.getDimension()
        fm.beginChange()
        self._dataProjectionCoordinateField = fm.createFieldEmbedded(self._modelCoordinateField,
                                                                     self._storedMeshLocationField)
        self._dataProjectionDistanceCoordinateField = fm.createFieldSubtract(self._dataProjectionCoordinateField,
                                                                             self._dataCoordinateField)
        self._dataProjectionErrorField = fm.createFieldMagnitude(self._dataProjectionDistanceCoordinateField)
        self._dataProjectionMeanErrorField = fm.createFieldNodesetMean(self._dataProjectionErrorField,
                                                                       activeDatapointsGroup)
        self._dataProjectionMaximumErrorField = fm.createFieldNodesetMaximum(self._dataProjectionErrorField,
                                                                             activeDatapointsGroup)

        nodetemplate = datapoints.createNodetemplate()
        nodetemplate.defineField(self._storedMeshLocationField)
        cache = fm.createFieldcache()
        dataIter = activeDatapointsGroup.createNodeiterator()
        datapoint = dataIter.next()
        while datapoint.isValid():
            cache.setNode(datapoint)
            element, xi = self._findMeshLocationField.evaluateMeshLocation(cache, dimension)
            if element.isValid():
                datapoint.merge(nodetemplate)
                self._storedMeshLocationField.assignMeshLocation(cache, element, xi)
            datapoint = dataIter.next()
        fm.endChange()
        self._showDataProjections()
        return

    def fit(self):
        fm = self._region.getFieldmodule()
        tmpTrue = fm.createFieldConstant([1])
        activeDatapointsGroup = self._activeDataPointGroupField.getNodesetGroup()
        activeDatapointsGroup.addNodesConditional(tmpTrue)
        optimisation = fm.createOptimisation()
        optimisation.setMethod(Optimisation.METHOD_LEAST_SQUARES_QUASI_NEWTON)
        objectiveFunction = fm.createFieldNodesetSumSquares(self._dataProjectionDistanceCoordinateField,
                                                            activeDatapointsGroup)
        result = optimisation.addObjectiveField(objectiveFunction)
        if result != ZINC_OK:
            raise ValueError('src.scaffoldfitter.fitter.Fitter.fit: Objective function was not initiated!'
                             'Ensure the projection is done successfully in order to calculate the distance.')
        numberOfGaussPoints = 4
        mesh = self.getMesh()

        lineMesh = fm.findMeshByDimension(1)
        if self._exteriorFaceMeshGroup is not None:
            lineMeshGroup = fm.createFieldElementGroup(lineMesh).getMeshGroup()
            if lineMeshGroup.isValid():
                lineMesh = lineMeshGroup

        if self.getFitStrainPenalty() > 0.0:
            strainField = self._getStrainField(mesh)
            weightField = fm.createFieldConstant(self.getFitStrainPenalty())
            weightedStrainField = strainField * weightField
            weightedStrainFieldIntegral = fm.createFieldMeshIntegralSquares(weightedStrainField,
                                                                            self._modelReferenceCoordinateField,
                                                                            mesh)
            weightedStrainFieldIntegral.setNumbersOfPoints(numberOfGaussPoints)
            result = optimisation.addObjectiveField(weightedStrainFieldIntegral)
            if result != ZINC_OK:
                raise ValueError('Could not add optimisation strain penalty objective field')

        # if self.getFitEdgeDiscontinuityPenalty() > 0.0:
        #     edgeDiscontinuityField = fm.createFieldEdgeDiscontinuity(self._modelCoordinateField)
        #     weightField = fm.createFieldConstant(self.getFitEdgeDiscontinuityPenalty())
        #     weightedEdgeDiscontinuityField = edgeDiscontinuityField*weightField
        #     weightedEdgeDiscontinuityIntegral = fm.createFieldMeshIntegralSquares(weightedEdgeDiscontinuityField,
        #                                                                           self._modelReferenceCoordinateField,
        #                                                                           lineMesh)
        #     weightedEdgeDiscontinuityIntegral.setNumbersOfPoints(numberOfGaussPoints)
        #     result = optimisation.addObjectiveField(weightedEdgeDiscontinuityIntegral)
        #     if result != ZINC_OK:
        #         raise ValueError('Could not add optimisation edge discontinuity penalty objective field')

        independentField = self._modelCoordinateField.castFiniteElement()
        result = optimisation.addIndependentField(independentField)
        if result != ZINC_OK:
            raise ValueError('Could not set optimisation dependent field')
        # if self._exteriorFaceGroup is not None:
            # optimisation.setConditionalField(independentField, self._exteriorSurfaceField)
        result = optimisation.setAttributeInteger(Optimisation.ATTRIBUTE_MAXIMUM_ITERATIONS, self.getFitMaxIterations())
        if result != ZINC_OK:
            raise ValueError('Could not set optimisation maximum iterations')

        loggerMessageCount = self._logger.getNumberOfMessages()

        result = optimisation.optimise()

        if loggerMessageCount > 0:
            for i in range(1, loggerMessageCount + 1):
                print(self._logger.getMessageTypeAtIndex(i), self._logger.getMessageTextAtIndex(i))
                self._logger.removeAllMessages()

        if result != ZINC_OK:
            raise ValueError('Optimisation failed with result ' + str(result))
        self._autorangeSpectrum()

    def _showDataProjections(self):
        self._hideDataProjections()
        scene = self._region.getScene()
        scene.beginChange()
        materialmodule = scene.getMaterialmodule()
        spectrummodule = scene.getSpectrummodule()
        defaultSpectrum = spectrummodule.getDefaultSpectrum()

        errorBars = scene.createGraphicsPoints()
        errorBars.setName('data-projections')
        errorBars.setFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        errorBars.setCoordinateField(self._dataCoordinateField)
        errorBars.setSubgroupField(self._activeDataPointGroupField)
        pointAttr = errorBars.getGraphicspointattributes()
        pointAttr.setGlyphShapeType(Glyph.SHAPE_TYPE_LINE)
        pointAttr.setBaseSize([0.0, 1.0, 1.0])
        pointAttr.setScaleFactors([1.0, 0.0, 0.0])
        pointAttr.setOrientationScaleField(self._dataProjectionDistanceCoordinateField)
        errorBars.setDataField(self._dataProjectionErrorField)
        errorBars.setSpectrum(defaultSpectrum)

        meanError = scene.createGraphicsPoints()
        meanError.setName('data-mean-error')
        meanError.setScenecoordinatesystem(SCENECOORDINATESYSTEM_NORMALISED_WINDOW_FIT_LEFT)
        pointAttr = meanError.getGraphicspointattributes()
        pointAttr.setBaseSize([1.0, 1.0, 1.0])
        pointAttr.setGlyphOffset([-0.9, 0.9, 0.0])
        pointAttr.setGlyphShapeType(Glyph.SHAPE_TYPE_NONE)
        pointAttr.setLabelText(1, 'Mean error:  ')
        pointAttr.setLabelField(self._dataProjectionMeanErrorField)

        maximumError = scene.createGraphicsPoints()
        maximumError.setName('data-maximum-error')
        maximumError.setScenecoordinatesystem(SCENECOORDINATESYSTEM_NORMALISED_WINDOW_FIT_LEFT)
        pointAttr = maximumError.getGraphicspointattributes()
        pointAttr.setBaseSize([1.0, 1.0, 1.0])
        pointAttr.setGlyphOffset([-0.9, 0.8, 0.0])
        pointAttr.setGlyphShapeType(Glyph.SHAPE_TYPE_NONE)
        pointAttr.setLabelText(1, 'Max. error:  ')
        pointAttr.setLabelField(self._dataProjectionMaximumErrorField)
        maximumError.setMaterial(materialmodule.findMaterialByName('red'))

        surfaces = scene.findGraphicsByName('fit-surfaces')
        scene.moveGraphicsBefore(surfaces, Graphics())

        self._autorangeSpectrum()
        scene.endChange()

    def _hideDataProjections(self):
        scene = self._region.getScene()
        scene.beginChange()
        graphics = scene.findGraphicsByName('data-projections')
        if graphics.isValid():
            scene.removeGraphics(graphics)
        graphics = scene.findGraphicsByName('data-mean-error')
        if graphics.isValid():
            scene.removeGraphics(graphics)
        graphics = scene.findGraphicsByName('data-maximum-error')
        if graphics.isValid():
            scene.removeGraphics(graphics)
        scene.endChange()

    def _autorangeSpectrum(self):
        scene = self._region.getScene()
        spectrummodule = scene.getSpectrummodule()
        spectrum = spectrummodule.getDefaultSpectrum()
        scenefiltermodule = scene.getScenefiltermodule()
        scenefilter = scenefiltermodule.getDefaultScenefilter()
        spectrum.autorange(scene, scenefilter)

    def _getModelCentre(self):
        mins, maxs = self._getModelRange()
        self._modelCentre = maths.mult(maths.add(mins, maxs), 0.5)
        return self._modelCentre

    def _getChildRegion(self):
        self._region = self._region.getFirstChild()

    def applyAlignSettings(self):
        rot = maths.eulerToRotationMatrix3(self._alignSettings['euler_angles'])
        scale = self._alignSettings['scale']
        if isinstance(scale, float):
            scale = [scale]*3
        elif not (isinstance(scale, list) and len(scale) == 3):
            scale = [1.0]*3
        xScale = scale
        if self.isAlignMirror():
            xScale = -scale
        rotationScale = [
            rot[0][0] * scale[0], rot[0][1] * scale[0], rot[0][2] * scale[0],
            rot[1][0] * scale[1], rot[1][1] * scale[1], rot[1][2] * scale[1],
            rot[2][0] * scale[2], rot[2][1] * scale[2], rot[2][2] * scale[2]]
        fm = self._region.getFieldmodule()
        fm.beginChange()
        if self._modelTransformedCoordinateField is None:
            self._modelRotationScaleField = fm.createFieldConstant(rotationScale)
            # following works in 3-D only
            temp1 = fm.createFieldMatrixMultiply(3, self._modelRotationScaleField, self._modelCoordinateField)
            self._modelOffsetField = fm.createFieldConstant(self._alignSettings['offset'])
            self._modelTransformedCoordinateField = fm.createFieldAdd(temp1, self._modelOffsetField)
        else:
            cache = fm.createFieldcache()
            self._modelRotationScaleField.assignReal(cache, rotationScale)
            self._modelOffsetField.assignReal(cache, self._alignSettings['offset'])
        fm.endChange()
        if not self._modelTransformedCoordinateField.isValid():
            print("Can't create transformed model coordinate field. Is problem 2-D?")
        self._alignSettingsChangeCallback()

    def setAlignSettingsChangeCallback(self, alignSettingsChangeCallback):
        self._alignSettingsChangeCallback = alignSettingsChangeCallback

    def setRegion(self):
        self._getChildRegion()

    def _setRefereceModelCoordinates(self, field):
        self._modelReferenceCoordinateField = field

    def _setModelCoordinates(self, field):
        self._modelCoordinateField = field

    def _setDataCoordinates(self, field):
        self._dataCoordinateField = field

    def setStatePostAlign(self):
        if not self._isStateAlign:
            return
        tmp = [self._alignSettings['scale'][0], self._alignSettings['scale'][2], self._alignSettings['scale'][1]]
        rotationScale = maths.matrixvectormult(maths.eulerToRotationMatrix3(self._alignSettings['euler_angles']),
                                                 tmp)
        if self.isAlignMirror():
            rotationScale[0] = maths.mult(rotationScale[0], -1.0)
        zincutils.transformCoordinates(self._modelCoordinateField, rotationScale, self._alignSettings['offset'])
        zincutils.copyNodalParameters(self._modelCoordinateField, self._modelReferenceCoordinateField)

    def _getNodesetMinimumMaximum(self, nodeset, field):
        fm = field.getFieldmodule()
        count = field.getNumberOfComponents()
        minimumsField = fm.createFieldNodesetMinimum(field, nodeset)
        maximumsField = fm.createFieldNodesetMaximum(field, nodeset)
        cache = fm.createFieldcache()
        result, minimums = minimumsField.evaluateReal(cache, count)
        if result != ZINC_OK:
            minimums = None
        result, maximums = maximumsField.evaluateReal(cache, count)
        if result != ZINC_OK:
            maximums = None
        del minimumsField
        del maximumsField
        return minimums, maximums

    def _getModelRange(self):
        fm = self._region.getFieldmodule()
        nodes = fm.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        minimums, maximums = self._getNodesetMinimumMaximum(nodes, self._modelReferenceCoordinateField)
        return minimums, maximums

    def _getScaffoldNodeParameters(self):
        parameterValues = zincutils.getScaffoldNodalParametersToList(self._modelCoordinateField)
        return parameterValues

    def _getPointCloudParameters(self):
        parameterValues = zincutils.getPointCloudParametersToList(self._dataCoordinateField)
        return parameterValues
