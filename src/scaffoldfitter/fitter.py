from opencmiss.zinc.field import Field, FieldFindMeshLocation
from opencmiss.zinc.status import OK as ZINC_OK

from .utils import maths


class Fitter(object):

    def __init__(self, region):
        self._region = region
        self.modelReferenceCoordinateField = None
        self.modelCoordinateField = None
        self._alignSettings = None
        self._modelTransformedCoordinateField = None
        self._modelRotationScaleField = None
        self._modelOffsetField = None

    def resetAlignSettings(self):
        self._alignSettings = dict(euler_angles=[0.0, 0.0, 0.0], scale=1.0, offset=[0.0, 0.0, 0.0], mirror=False)

    def isAlignMirror(self):
        return self._alignSettings['mirror']

    def createFiniteElementField(self, fieldName='coordinates'):
        fieldModule = self._region.getFieldmodule()
        fieldModule.beginChange()

        finiteElementField = fieldModule.createFieldFiniteElement(3)

        finiteElementField.setName(fieldName)

        finiteElementField.setManaged(True)
        finiteElementField.setTypeCoordinate(True)
        fieldModule.endChange()

        return finiteElementField

    def getMesh(self):
        fm = self._region.getFieldmodule()
        for dimension in range(3, 0, -1):
            mesh = fm.findMeshByDimension(dimension)
            if mesh.getSize() > 0:
                return mesh
        raise ValueError('Model contains no mesh')

    def getModelCoordinateField(self):
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
                     (self.modelReferenceCoordinateField != field)):
                if field.isDefinedAtLocation(cache):
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
            if field.isTypeCoordinate() and (field.getNumberOfComponents() <= 3) and\
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

    def getModelCentre(self):
        mins, maxs = self._getModelRange()
        return maths.mult(maths.add(mins, maxs), 0.5)

    def applyAlignSettings(self):
        rot = maths.eulerToRotationMatrix3(self._alignSettings['euler_angles'])
        scale = self._alignSettings['scale']
        xScale = scale
        if self.isAlignMirror():
            xScale = -scale
        rotationScale = [
            rot[0][0]*xScale, rot[0][1]*xScale, rot[0][2]*xScale,
            rot[1][0]*scale,  rot[1][1]*scale,  rot[1][2]*scale,
            rot[2][0]*scale,  rot[2][1]*scale,  rot[2][2]*scale]
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
        minimums, maximums = self._getNodesetMinimumMaximum(nodes, self.modelCoordinateField)
        return minimums, maximums
