'''
Utility functions for easing use of Zinc API.
'''

from opencmiss.zinc.context import Context
from opencmiss.zinc.field import Field
from opencmiss.zinc.fieldmodule import Fieldmodule
from opencmiss.zinc.node import Node, Nodeset
from opencmiss.zinc.result import RESULT_OK

def assignFieldParameters(targetField : Field, sourceField : Field):
    """
    Copy parameters from sourceField to targetField.
    Currently only works for node parameters.
    """
    fieldassignment = targetField.createFieldassignment(sourceField)
    fieldassignment.assign()

def createFieldClone(sourceField : Field, targetName):
    """
    Copy an existing field to a new field of supplied name.
    :param sourceField: Zinc finite element field to copy.
    :param targetName: The name of the new field, assumed different from that of source.
    :return: New identically defined field with supplied name.
    """
    assert sourceField.castFiniteElement().isValid(), 'createFieldClone. Not a Zinc finite element field'
    fieldmodule = sourceField.getFieldmodule()
    fieldmodule.beginChange()
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
    fieldmodule.endChange()  # currently must do this before field can be found
    #region.writeFile("C:\\users\\gchr006\\tmp\\km.exf")
    field = fieldmodule.findFieldByName(targetName).castFiniteElement()
    assert field.isValid()
    return field

def createFieldEulerAnglesRotationMatrix(fieldmodule : Fieldmodule, eulerAngles : Field):
    """
    From OpenCMISS-Zinc graphics_library.cpp, transposed.
    :param eulerAngles: 3-component field of angles in radians, components:
    1 = azimuth (about z)
    2 = elevation (about rotated y)
    3 = roll (about rotated x)
    :return: 3x3 rotation matrix field suitable for pre-multiplying [x, y, z].
    """
    assert eulerAngles.getNumberOfComponents() == 3
    fieldmodule.beginChange()
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
        fieldmodule.createFieldMultiply(cos_azimuth, cos_elevation),
        fieldmodule.createFieldSubtract(
            fieldmodule.createFieldMultiply(cos_azimuth_sin_elevation, sin_roll),
            fieldmodule.createFieldMultiply(sin_azimuth, cos_roll)),
        fieldmodule.createFieldAdd(
            fieldmodule.createFieldMultiply(cos_azimuth_sin_elevation, cos_roll),
            fieldmodule.createFieldMultiply(sin_azimuth, sin_roll)),
        fieldmodule.createFieldMultiply(sin_azimuth, cos_elevation),
        fieldmodule.createFieldAdd(
            fieldmodule.createFieldMultiply(sin_azimuth_sin_elevation, sin_roll),
            fieldmodule.createFieldMultiply(cos_azimuth, cos_roll)),
        fieldmodule.createFieldSubtract(
            fieldmodule.createFieldMultiply(sin_azimuth_sin_elevation, cos_roll),
            fieldmodule.createFieldMultiply(cos_azimuth, sin_roll)),
        fieldmodule.createFieldMultiply(minus_one, sin_elevation),
        fieldmodule.createFieldMultiply(cos_elevation, sin_roll),
        fieldmodule.createFieldMultiply(cos_elevation, cos_roll) ]
    rotationMatrix = fieldmodule.createFieldConcatenate(matrixComponents)
    fieldmodule.endChange()
    return rotationMatrix

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

def getNodeLabelCentres(nodeset : Nodeset, coordinatesField : Field, labelField : Field):
    """
    Find mean locations of node coordinate with the same labels.
    :param nodeset: Zinc Nodeset or NodesetGroup to search.
    :param coordinatesField: The coordinate field to evaluate.
    :param labelField: The label field to match.
    :return: Dict of labels -> coordinates.
    """
    componentsCount = coordinatesField.getNumberOfComponents()
    fieldmodule = nodeset.getFieldmodule()
    fieldcache = fieldmodule.createFieldcache()
    labelRecords = {}  # label -> (coordinates, count)
    nodeiter = nodeset.createNodeiterator()
    node = nodeiter.next()
    while node.isValid():
        fieldcache.setNode(node)
        label = labelField.evaluateString(fieldcache)
        coordinatesResult, coordinates = coordinatesField.evaluateReal(fieldcache, componentsCount)
        if label and (coordinatesResult == RESULT_OK):
            labelRecord = labelRecords.get(label)
            if labelRecord:
                labelCentre = labelRecord[0]
                for c in range(componentsCount):
                    labelCentre[c] += coordinates[c]
                labelRecord[1] += 1
            else:
                labelRecords[label] = (coordinates, 1)
        node = nodeiter.next()
    # divide centre coordinates by count
    labelCentres = {}
    for label in labelRecords:
        labelRecord = labelRecords[label]
        labelCount = labelRecord[1]
        labelCentre = labelRecord[0]
        if labelCount > 1:
            scale = 1.0/labelCount
            for c in range(componentsCount):
                labelCentre[c] *= scale
        labelCentres[label] = labelCentre
    return labelCentres

def evaluateNodesetMeanCoordinates(nodeset, coordinates):
    fieldmodule = nodeset.getFieldmodule()
    componentsCount = coordinates.getNumberOfComponents()
    fieldmodule.beginChange()
    meanCoordinates = fieldmodule.createFieldNodesetMean(coordinates, nodeset)
    fieldcache = fieldmodule.createFieldcache()
    result, centre = meanCoordinates.evaluateReal(fieldcache, componentsCount)
    del meanCoordinates
    del fieldcache
    fieldmodule.endChange()
    assert result == RESULT_OK
    return centre

def getUniqueFieldName(fieldmodule : Fieldmodule, stemName):
    """
    Return an unique field name formed by stemName plus a number,
    not in use by any field in fieldmodule.
    """
    number = 1
    while True:
        fieldName = stemName + str(number)
        field = fieldmodule.findFieldByName(fieldName)
        if not field.isValid():
            return fieldname
        number += 1
