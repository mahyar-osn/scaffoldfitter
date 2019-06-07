from opencmiss.zinc.field import Field, FieldFindMeshLocation
from opencmiss.zinc.status import OK as ZINC_OK

from .utils import maths
from .utils import zincutils
from .optimization.alignment_fitting import fitRigidScale


class Fitter(object):

    def __init__(self, region):
        self._region = region
        self.modelReferenceCoordinateField = None
        self.modelCoordinateField = None
        self._alignSettings = None
        self._modelTransformedCoordinateField = None
        self._modelRotationScaleField = None
        self._modelOffsetField = None
        self._dataCoordinateField = None
        self._modelCentre = None
        self._isStateAlign = True

    def resetAlignSettings(self):
        self._alignSettings = dict(euler_angles=[0.0, 0.0, 0.0], scale=1.0, offset=[0.0, 0.0, 0.0], mirror=False)

    def getInitialScale(self):
        dataMinimums, dataMaximums = self._getDataRange()
        dataRange = maths.sub(dataMaximums, dataMinimums)
        modelMinimums, modelMaximums = self._getModelRange()
        modelRange = maths.sub(modelMaximums, modelMinimums)
        dataModelDifference = maths.eldiv(dataRange, modelRange)
        meanScale = sum(dataModelDifference)/len(dataModelDifference)
        self.setAlignScale(int(meanScale))
        self.setStatePostAlign()
        self.resetAlignSettings()

    def autoCentreModelOnData(self):
        dataMinimums, dataMaximums = self._getDataRange()
        dataCentre = maths.mult(maths.add(dataMinimums, dataMaximums), 0.5)
        modelMinimums, modelMaximums = self._getModelRange()
        modelCentre = maths.mult(maths.add(modelMinimums, modelMaximums), 0.5)
        self.setAlignOffset(maths.sub(dataCentre, modelCentre))
        self.setStatePostAlign()
        return self.modelCoordinateField

    def initializeRigidAlignment(self):
        scaffoldNodeValus = self._getScaffoldNodeParameters()
        pointCloudValues = self._getPointCloudParameters()
        t0pt, fitted_data, (initial_rms, final_rms), T = fitRigidScale(scaffoldNodeValus, pointCloudValues,
                                                                         xtol=1e-10, maxfev=0, sample=None,
                                                                         scale_threshold=None, output_errors=True)

        print("Initial RMS = ", initial_rms)
        print("Final RMS = ", final_rms)

        euler_radians = maths.rotationMatrix3ToEuler(T[:3, :3].tolist())
        euler_radians_correct = [euler_radians[0], euler_radians[1], euler_radians[2]]
        self.setAlignEulerAngles(euler_radians_correct)
        self.setAlignScale(1.)
        self.setStatePostAlign()
        self._isStateAlign = False
        return self.modelCoordinateField

    def swap_axes(self, axes=None):
        zincutils.swap_axes(self.modelCoordinateField, axes=axes)
        # zincutils.copyNodalParameters(self.modelCoordinateField, self.modelReferenceCoordinateField)

    def isAlignMirror(self):
        return self._alignSettings['mirror']

    def getAlignEulerAngles(self):
        return self._alignSettings['euler_angles']

    def getAlignScale(self):
        return self._alignSettings['scale']

    def getAlignOffset(self):
        return self._alignSettings['offset']

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

    def createFiniteElementField(self, fieldName='coordinates'):
        fieldModule = self._region.getFieldmodule()
        fieldModule.beginChange()
        finiteElementField = fieldModule.createFieldFiniteElement(3)
        finiteElementField.setName(fieldName)
        finiteElementField.setManaged(True)
        finiteElementField.setTypeCoordinate(True)
        fieldModule.endChange()
        return finiteElementField

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
                    ((self.modelReferenceCoordinateField is None) or
                     (self.modelReferenceCoordinateField != field) or
                     (self.modelCoordinateField is None) or
                     (self.modelCoordinateField != field)):
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
        datapoint = datapoints.createNodeiterator().next()
        if not datapoint.isValid():
            raise ValueError('Point cloud is empty')
        cache = fm.createFieldcache()
        cache.setNode(datapoint)
        fieldIter = fm.createFielditerator()
        field = fieldIter.next()
        while field.isValid():
            if field.isTypeCoordinate() and (field.getNumberOfComponents() <= 3) and \
                    ((self.modelReferenceCoordinateField is None) or (field != self.modelReferenceCoordinateField)):
                if field.isDefinedAtLocation(cache):
                    return field
            field = fieldIter.next()
        raise ValueError('Could not determine data coordinate field')

    def getProjectSurfaceGroup(self):
        fm = self._region.getFieldmodule()
        projectSurfaceGroup = fm.findFieldByName('projectsurface').castGroup()
        if projectSurfaceGroup.isValid():
            mesh2d = fm.findMeshByDimension(2)
            projectSurfaceElementGroup = projectSurfaceGroup.getFieldElementGroup(mesh2d)
            if projectSurfaceElementGroup.isValid() and (projectSurfaceElementGroup.getMeshGroup().getSize() > 0):
                return projectSurfaceGroup, projectSurfaceElementGroup
        return None, None

    def _getModelCentre(self):
        mins, maxs = self._getModelRange()
        self._modelCentre = maths.mult(maths.add(mins, maxs), 0.5)
        return self._modelCentre

    def _getChildRegion(self):
        self._region = self._region.getFirstChild()

    def applyAlignSettings(self):
        rot = maths.eulerToRotationMatrix3(self._alignSettings['euler_angles'])
        scale = self._alignSettings['scale']
        xScale = scale
        if self.isAlignMirror():
            xScale = -scale
        rotationScale = [
            rot[0][0] * xScale, rot[0][1] * xScale, rot[0][2] * xScale,
            rot[1][0] * scale, rot[1][1] * scale, rot[1][2] * scale,
            rot[2][0] * scale, rot[2][1] * scale, rot[2][2] * scale]
        fm = self._region.getFieldmodule()
        fm.beginChange()
        if self._modelTransformedCoordinateField is None:
            self._modelRotationScaleField = fm.createFieldConstant(rotationScale)
            # following works in 3-D only
            temp1 = fm.createFieldMatrixMultiply(3, self._modelRotationScaleField, self.modelCoordinateField)
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
        self.modelReferenceCoordinateField = field

    def _setModelCoordinates(self, field):
        self.modelCoordinateField = field

    def setStatePostAlign(self):
        if not self._isStateAlign:
            return
        rotationScale = maths.matrixconstantmult(maths.eulerToRotationMatrix3(self._alignSettings['euler_angles']),
                                                 self._alignSettings['scale'])
        if self.isAlignMirror():
            rotationScale[0] = maths.mult(rotationScale[0], -1.0)
        zincutils.transformCoordinates(self.modelCoordinateField, rotationScale, self._alignSettings['offset'])
        # zincutils.copyNodalParameters(self.modelCoordinateField, self.modelReferenceCoordinateField)

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
        minimums, maximums = self._getNodesetMinimumMaximum(nodes, self.modelReferenceCoordinateField)
        return minimums, maximums

    def _getScaffoldNodeParameters(self):
        # parameterValues = zincutils.getScaffoldNodalParametersToList(self.modelReferenceCoordinateField)
        parameterValues = zincutils.getScaffoldNodalParametersToList(self.modelCoordinateField)
        return parameterValues

    def _getPointCloudParameters(self):
        parameterValues = zincutils.getPointCloudParametersToList(self._dataCoordinateField)
        return parameterValues
