"""
Utility functions for easing use of Zinc API.
"""

from opencmiss.zinc.context import Context
from opencmiss.zinc.field import Field
from opencmiss.zinc.fieldmodule import Fieldmodule
from opencmiss.zinc.element import Mesh
from opencmiss.zinc.node import Node, Nodeset
from opencmiss.zinc.result import RESULT_OK

class ZincCacheChanges:
    """
    Context manager for ensuring beginChange, endChange always called on
    supplied object, even with exceptions.
    Usage:
    with ZincCacheChanges(object):
        # make multiple changes to object or objects it owns
    """

    def __init__(self, object):
        """
        :param object: Zinc object with beginChange/endChange methods.
        """
        self._object = object

    def __enter__(self):
        self._object.beginChange()
        return self

    def __exit__(self, *args):
        self._object.endChange()


def assignFieldParameters(targetField : Field, sourceField : Field):
    """
    Copy parameters from sourceField to targetField.
    Currently only works for node parameters.
    """
    fieldassignment = targetField.createFieldassignment(sourceField)
    fieldassignment.assign()

def createFieldClone(sourceField : Field, targetName : str, managed=False) -> Field:
    """
    Copy an existing field to a new field of supplied name.
    :param sourceField: Zinc finite element field to copy.
    :param targetName: The name of the new field, assumed different from that of source.
    :return: New identically defined field with supplied name.
    """
    assert sourceField.castFiniteElement().isValid(), "createFieldClone. Not a Zinc finite element field"
    fieldmodule = sourceField.getFieldmodule()
    with ZincCacheChanges(fieldmodule):
        field = fieldmodule.findFieldByName(targetName)
        if field.isValid():
            field.setName(getUniqueFieldName(fieldmodule, "destroy_" + targetName))
            field.setManaged(False)
        # Zinc needs a function to do this efficiently; currently serialise to string, replace field name and reload!
        sourceName = sourceField.getName()
        region = fieldmodule.getRegion()
        sir = region.createStreaminformationRegion()
        srm = sir.createStreamresourceMemory()
        sir.setFieldNames([ sourceName ])
        region.write(sir)
        result, buffer = srm.getBuffer()
        # small risk of modifying other text here:
        sourceBytes = bytes(") " + sourceName + ",", "utf-8")
        targetBytes = bytes(") " + targetName + ",", "utf-8")
        buffer = buffer.replace(sourceBytes, targetBytes)
        sir = region.createStreaminformationRegion()
        srm = sir.createStreamresourceMemoryBuffer(buffer)
        result = region.read(sir)
        assert result == RESULT_OK
    # note currently must have called endChange before field can be found
    field = fieldmodule.findFieldByName(targetName).castFiniteElement()
    field.setManaged(managed)
    assert field.isValid()
    return field

def createFieldEulerAnglesRotationMatrix(fieldmodule : Fieldmodule, eulerAngles : Field) -> Field:
    """
    From OpenCMISS-Zinc graphics_library.cpp, transposed.
    :param eulerAngles: 3-component field of angles in radians, components:
    1 = azimuth (about z)
    2 = elevation (about rotated y)
    3 = roll (about rotated x)
    :return: 3x3 rotation matrix field suitable for pre-multiplying [x, y, z].
    """
    assert eulerAngles.getNumberOfComponents() == 3
    with ZincCacheChanges(fieldmodule):
        azimuth = fieldmodule.createFieldComponent(eulerAngles, 1)
        cos_azimuth = fieldmodule.createFieldCos(azimuth)
        sin_azimuth = fieldmodule.createFieldSin(azimuth)
        elevation = fieldmodule.createFieldComponent(eulerAngles, 2)
        cos_elevation = fieldmodule.createFieldCos(elevation)
        sin_elevation = fieldmodule.createFieldSin(elevation)
        roll = fieldmodule.createFieldComponent(eulerAngles, 3)
        cos_roll = fieldmodule.createFieldCos(roll)
        sin_roll = fieldmodule.createFieldSin(roll)
        minus_one = fieldmodule.createFieldConstant([ -1.0 ])
        cos_azimuth_sin_elevation = fieldmodule.createFieldMultiply(cos_azimuth, sin_elevation)
        sin_azimuth_sin_elevation = fieldmodule.createFieldMultiply(sin_azimuth, sin_elevation)
        matrixComponents = [
            cos_azimuth*cos_elevation,
            cos_azimuth_sin_elevation*sin_roll - sin_azimuth*cos_roll,
            cos_azimuth_sin_elevation*cos_roll + sin_azimuth*sin_roll,
            sin_azimuth*cos_elevation,
            sin_azimuth_sin_elevation*sin_roll + cos_azimuth*cos_roll,
            sin_azimuth_sin_elevation*cos_roll - cos_azimuth*sin_roll,
            minus_one*sin_elevation,
            cos_elevation*sin_roll,
            cos_elevation*cos_roll ]
        rotationMatrix = fieldmodule.createFieldConcatenate(matrixComponents)
    return rotationMatrix

def getOrCreateFieldFiniteElement(fieldmodule : Fieldmodule, fieldName, componentsCount=3, componentNames=None, managed=False) -> Field:
    """
    Finds or creates a finite element field for the specified number of real components.
    Asserts existing field is finite element type withcorrect attributes.
    :param fieldmodule:  Zinc Fieldmodule to find or create field in.
    :param fieldName:  Name of field to find or create.
    :param componentsCount: Number of components / dimension of field, from 1 to 3.
    :param componentNames: Optional list of component names.
    :param managed: Managed state of field if created here.
    :return: Zinc Field.
    """
    assert (componentsCount > 0), "getOrCreateFieldFiniteElement.  Invalid componentsCount"
    assert (not componentNames) or (len(componentNames) == componentsCount), "getOrCreateRealField.  Invalid componentNames"
    field = fieldmodule.findFieldByName(fieldName)
    if field.isValid():
        field = field.castFiniteElement()
        assert field.isValid(), "getOrCreateFieldFiniteElement.  Existing field " + fieldName + " is not finite element type"
        assert field.getNumberOfComponents() == componentsCount, "getOrCreateFieldFiniteElement.  Existing field " + fieldName + " does not have " + str(componentsCount) + " components"
        return field
    with ZincCacheChanges(fieldmodule):
        field = fieldmodule.createFieldFiniteElement(componentsCount)
        field.setName(fieldName)
        field.setManaged(managed)
        #field.setTypeCoordinate(True)
        if componentNames:
            for c in range(componentsCount):
                field.setComponentName(c + 1, componentNames[c])
    return field

def createDisplacementGradientFields(coordinates : Field, referenceCoordinates : Field, mesh : Mesh):
    """
    :return: 1st and 2nd displacement gradients of (coordinates - referenceCoordinates) w.r.t. referenceCoordinates.
    """
    assert (coordinates.getNumberOfComponents() == 3) and (referenceCoordinates.getNumberOfComponents() == 3)
    fieldmodule = mesh.getFieldmodule()
    dimension = mesh.getDimension()
    displacementGradient = None
    displacementGradient2 = None
    with ZincCacheChanges(fieldmodule):
        if dimension == 3:
            u = coordinates  - referenceCoordinates
            displacementGradient = fieldmodule.createFieldGradient(u, referenceCoordinates)
            displacementGradient2 = fieldmodule.createFieldGradient(displacementGradient, referenceCoordinates)
        elif dimension == 2:
            # assume xi directions are approximately normal; effect is to penalise elements where this is not so, which is also desired
            dX_dxi1 = fieldmodule.createFieldDerivative(referenceCoordinates, 1)
            dX_dxi2 = fieldmodule.createFieldDerivative(referenceCoordinates, 2)
            dx_dxi1 = fieldmodule.createFieldDerivative(coordinates, 1)
            dx_dxi2 = fieldmodule.createFieldDerivative(coordinates, 2)
            dS1_dxi1 = fieldmodule.createFieldMagnitude(dX_dxi1)
            dS2_dxi2 = fieldmodule.createFieldMagnitude(dX_dxi2)
            du_dS1 = (dx_dxi1 - dX_dxi1)/dS1_dxi1
            du_dS2 = (dx_dxi2 - dX_dxi2)/dS2_dxi2
            displacementGradient = fieldmodule.createFieldConcatenate([du_dS1, du_dS2])
            # curvature:
            d2u_dSdxi1 = fieldmodule.createFieldDerivative(displacementGradient, 1)
            d2u_dSdxi2 = fieldmodule.createFieldDerivative(displacementGradient, 2)
            displacementGradient2 = fieldmodule.createFieldConcatenate([ d2u_dSdxi1/dS1_dxi1, d2u_dSdxi2/dS2_dxi2 ])
        else:  # dimension == 1
            dX_dxi1 = fieldmodule.createFieldDerivative(referenceCoordinates, 1)
            dx_dxi1 = fieldmodule.createFieldDerivative(coordinates, 1)
            dS1_dxi1 = fieldmodule.createFieldMagnitude(dX_dxi1)
            displacementGradient = (dx_dxi1 - dX_dxi1)/dS1_dxi1
            # curvature:
            displacementGradient2 = fieldmodule.createFieldDerivative(displacementGradient, 1)/dS1_dxi1
    return displacementGradient, displacementGradient2

def getOrCreateFieldMeshLocation(fieldmodule : Fieldmodule, mesh : Mesh, namePrefix = "location_") -> Field:
    """
    Get or create a stored mesh location field for storing locations in the
    supplied mesh, used for storing data projections.
    Note can't currently verify existing field stores locations in the supplied mesh.
    :param fieldmodule:  Zinc fieldmodule to find or create field in.
    :param mesh:  Mesh to store locations in, from same fieldmodule.
    :param namePrefix:  Prefix of field name. Function appends mesh name and
    possibly a unique number.
    """
    baseName = name = namePrefix + mesh.getName()
    number = 1
    while True:
        field = fieldmodule.findFieldByName(name)
        if not field.isValid():
            break
        meshLocationField = field.castStoredMeshLocation()
        if meshLocationField.isValid():
            return meshLocationField
        name = baseName + str(number)
        number += 1
    with ZincCacheChanges(fieldmodule):
        meshLocationField = fieldmodule.createFieldStoredMeshLocation(mesh)
        meshLocationField.setName(name)
        meshLocationField.setManaged(True)
    return meshLocationField

def createTransformationFields(coordinates : Field, rotationAngles = [ 0.0, 0.0, 0.0 ], scaleValue : float = 1.0, translationOffsets = [ 0.0, 0.0, 0.0 ]):
    """
    Create constant fields for rotation, scale and translation containing the supplied
    values, plus the transformed coordinates applying them in the supplied order.
    :param coordinates: The coordinate field to scale, 3 components.
    :param rotationAngles: List of euler angles, length = number of components. See createFieldEulerAnglesRotationMatrix.
    :param scaleValue: Scalar to multiply all components of coordinates.
    :param translationOffsets: List of offsets, length = number of components.
    :return: 4 fields: transformedCoordinates, rotation, scale, translation
    """
    componentsCount = coordinates.getNumberOfComponents()
    assert (componentsCount == 3) and (len(rotationAngles) == componentsCount) and isinstance(scaleValue, float) \
        and (len(translationOffsets) == componentsCount), "createTransformationFields.  Invalid arguments"
    fieldmodule = coordinates.getFieldmodule()
    with ZincCacheChanges(fieldmodule):
        # scale, translate and rotate model, in that order
        rotation = fieldmodule.createFieldConstant(rotationAngles)
        scale = fieldmodule.createFieldConstant(scaleValue)
        translation = fieldmodule.createFieldConstant(translationOffsets)
        rotationMatrix = createFieldEulerAnglesRotationMatrix(fieldmodule, rotation)
        rotatedCoordinates = fieldmodule.createFieldMatrixMultiply(componentsCount, rotationMatrix, coordinates)
        transformedCoordinates = rotatedCoordinates*scale + translation
        assert transformedCoordinates.isValid()
    return transformedCoordinates, rotation, scale, translation

def getGroupList(fieldmodule):
    """
    Get list of Zinc groups in fieldmodule.
    """
    groups = []
    fielditer = fieldmodule.createFielditerator()
    field = fielditer.next()
    while field.isValid():
        group = field.castGroup()
        if group.isValid():
            groups.append(group)
        field = fielditer.next()
    return groups

def findNodeWithName(nodeset : Nodeset, nameField : Field, name):
    """
    Get single node in nodeset with supplied name.
    :param nodeset: Zinc Nodeset or NodesetGroup to search.
    :param nameField: The name field to match.
    :param name: The name to match in nameField.
    :return: Node with name, or None if 0 or multiple nodes with name.
    """
    fieldmodule = nodeset.getFieldmodule()
    fieldcache = fieldmodule.createFieldcache()
    nodeiter = nodeset.createNodeiterator()
    nodeWithName = None
    node = nodeiter.next()
    while node.isValid():
        fieldcache.setNode(node)
        tempName = nameField.evaluateString(fieldcache)
        if tempName == name:
            if nodeWithName:
                return None
            nodeWithName = node
        node = nodeiter.next()
    return nodeWithName

def getNodeNameCentres(nodeset : Nodeset, coordinatesField : Field, nameField : Field):
    """
    Find mean locations of node coordinate with the same names.
    :param nodeset: Zinc Nodeset or NodesetGroup to search.
    :param coordinatesField: The coordinate field to evaluate.
    :param nameField: The name field to match.
    :return: Dict of names -> coordinates.
    """
    componentsCount = coordinatesField.getNumberOfComponents()
    fieldmodule = nodeset.getFieldmodule()
    fieldcache = fieldmodule.createFieldcache()
    nameRecords = {}  # name -> (coordinates, count)
    nodeiter = nodeset.createNodeiterator()
    node = nodeiter.next()
    while node.isValid():
        fieldcache.setNode(node)
        name = nameField.evaluateString(fieldcache)
        coordinatesResult, coordinates = coordinatesField.evaluateReal(fieldcache, componentsCount)
        if name and (coordinatesResult == RESULT_OK):
            nameRecord = nameRecords.get(name)
            if nameRecord:
                nameCentre = nameRecord[0]
                for c in range(componentsCount):
                    nameCentre[c] += coordinates[c]
                nameRecord[1] += 1
            else:
                nameRecords[name] = (coordinates, 1)
        node = nodeiter.next()
    # divide centre coordinates by count
    nameCentres = {}
    for name in nameRecords:
        nameRecord = nameRecords[name]
        nameCount = nameRecord[1]
        nameCentre = nameRecord[0]
        if nameCount > 1:
            scale = 1.0/nameCount
            for c in range(componentsCount):
                nameCentre[c] *= scale
        nameCentres[name] = nameCentre
    return nameCentres

def evaluateNodesetMeanCoordinates(coordinates : Field, nodeset : Nodeset):
    """
    :return: Mean of coordinates over nodeset.
    """
    fieldmodule = nodeset.getFieldmodule()
    componentsCount = coordinates.getNumberOfComponents()
    with ZincCacheChanges(fieldmodule):
        meanCoordinatesField = fieldmodule.createFieldNodesetMean(coordinates, nodeset)
        fieldcache = fieldmodule.createFieldcache()
        result, meanCoordinates = meanCoordinatesField.evaluateReal(fieldcache, componentsCount)
        assert result == RESULT_OK
        del meanCoordinatesField
        del fieldcache
    assert result == RESULT_OK
    return meanCoordinates

def evaluateNodesetCoordinatesRange(coordinates : Field, nodeset : Nodeset):
    """
    :return: min, max range of coordinates field over nodes.
    """
    fieldmodule = nodeset.getFieldmodule()
    componentsCount = coordinates.getNumberOfComponents()
    with ZincCacheChanges(fieldmodule):
        minCoordinates = fieldmodule.createFieldNodesetMinimum(coordinates, nodeset)
        maxCoordinates = fieldmodule.createFieldNodesetMaximum(coordinates, nodeset)
        fieldcache = fieldmodule.createFieldcache()
        result, minX = minCoordinates.evaluateReal(fieldcache, componentsCount)
        assert result == RESULT_OK
        result, maxX = maxCoordinates.evaluateReal(fieldcache, componentsCount)
        assert result == RESULT_OK
        del minCoordinates
        del maxCoordinates
        del fieldcache
    return minX, maxX

def createMeshVolumeField(coordinates : Field, mesh : Mesh, numberOfPoints = 3):
    """
    :param numberOfPoints: Number of Gauss points.
    :return: Field giving volume of coordinates field over mesh via Gaussian quadrature.
    """
    fieldmodule = coordinates.getFieldmodule()
    with ZincCacheChanges(fieldmodule):
        volumeField = fieldmodule.createFieldMeshIntegral(fieldmodule.createFieldConstant(1.0), coordinates, mesh)
        volumeField.setNumbersOfPoints(numberOfPoints)
    return volumeField

def getUniqueFieldName(fieldmodule : Fieldmodule, stemName : str) -> str:
    """
    Return a unique field name in fieldmodule either equal to stemName or
    appending a number starting at 1 and increasing.
    """
    field = fieldmodule.findFieldByName(stemName)
    if not field.isValid():
        return stemName
    number = 1
    while True:
        fieldName = stemName + str(number)
        field = fieldmodule.findFieldByName(fieldName)
        if not field.isValid():
            return fieldName
        number += 1

def FieldIsManagedCoordinates(field : Field):
    """
    Conditional function returning True if the field is Finite Element
    type with 3 components, and is managed.
    """
    return field.castFiniteElement().isValid() and (field.getNumberOfComponents() == 3) and field.isManaged()

def FieldIsManagedGroup(field : Field):
    """
    Conditional function returning True if the field is a managed Group.
    """
    return field.castGroup().isValid() and field.isManaged()
