from opencmiss.zinc.node import Node
from opencmiss.zinc.field import Field
from opencmiss.zinc.status import OK as ZINC_OK

from .maths import matrixvectormult, add, elmult


def copyNodalParameters(sourceField, targetField, time = 0.0):
    """
    Copy nodal parameters from sourceField to identically defined targetField.
    Assumes they are in the same field module.
    :param sourceField: the field to copy from
    :param targetField: the field to copy to
    :param optional time
    :return: True on success, otherwise false
    """
    ncomp = sourceField.getNumberOfComponents()
    if targetField.getNumberOfComponents() != ncomp:
        print('zinc.copyNodalParameters: fields must have same number of components')
        return False
    sourceFeField = sourceField.castFiniteElement()
    targetFeField = targetField.castFiniteElement()
    if not (sourceFeField.isValid() and targetFeField.isValid()):
        print('zinc.copyNodalParameters: fields must be finite element type')
        return False
    success = True
    fm = sourceFeField.getFieldmodule()
    fm.beginChange()
    cache = fm.createFieldcache()
    cache.setTime(time)
    nodes = fm.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
    nodetemplate = nodes.createNodetemplate()
    nodeIter = nodes.createNodeiterator()
    node = nodeIter.next()
    while node.isValid():
        nodetemplate.defineFieldFromNode(sourceFeField, node)
        cache.setNode(node)
        for derivative in [Node.VALUE_LABEL_VALUE, Node.VALUE_LABEL_D_DS1, Node.VALUE_LABEL_D_DS2, Node.VALUE_LABEL_D2_DS1DS2,
                           Node.VALUE_LABEL_D_DS3, Node.VALUE_LABEL_D2_DS1DS3, Node.VALUE_LABEL_D2_DS2DS3, Node.VALUE_LABEL_D3_DS1DS2DS3]:
            versions = nodetemplate.getValueNumberOfVersions(sourceFeField, -1, derivative)
            for v in range(1, versions + 1):
                result, values = sourceFeField.getNodeParameters(cache, -1, derivative, v, ncomp)
                if result != ZINC_OK:
                    success = False
                else:
                    result = targetFeField.setNodeParameters(cache, -1, derivative, v, values)
                    if result != ZINC_OK:
                        success = False
        node = nodeIter.next()
    fm.endChange()
    if not success:
        print('zinc.copyNodalParameters: failed to get/set some values')
    return success


def transformCoordinates(field, rotationScale, offset, time = 0.0):
    """
    Transform finite element field coordinates by matrix and offset, handling nodal derivatives and versions.
    Limited to nodal parameters, rectangular cartesian coordinates
    :param field: the coordinate field to transform
    :param rotationScale: square transformation matrix 2-D array with as many rows and columns as field components.
    :param offset: coordinates offset
    :return: True on success, otherwise false
    """
    ncomp = field.getNumberOfComponents()
    if ((ncomp != 2) and (ncomp != 3)):
        print('zinc.transformCoordinates: field has invalid number of components')
        return False
    if (len(rotationScale) != ncomp) or (len(offset) != ncomp):
        print('zinc.transformCoordinates: invalid matrix number of columns or offset size')
        return False
    # for matRow in rotationScale:
    #     if len(matRow) != ncomp:
    #         print('zinc.transformCoordinates: invalid matrix number of columns')
    #         return False
    if (field.getCoordinateSystemType() != Field.COORDINATE_SYSTEM_TYPE_RECTANGULAR_CARTESIAN):
        print('zinc.transformCoordinates: field is not rectangular cartesian')
        return False
    feField = field.castFiniteElement()
    if not feField.isValid():
        print('zinc.transformCoordinates: field is not finite element field type')
        return False
    success = True
    fm = field.getFieldmodule()
    fm.beginChange()
    cache = fm.createFieldcache()
    cache.setTime(time)
    nodes = fm.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
    nodetemplate = nodes.createNodetemplate()
    nodeIter = nodes.createNodeiterator()
    node = nodeIter.next()
    while node.isValid():
        nodetemplate.defineFieldFromNode(feField, node)
        cache.setNode(node)
        for derivative in [Node.VALUE_LABEL_VALUE, Node.VALUE_LABEL_D_DS1, Node.VALUE_LABEL_D_DS2, Node.VALUE_LABEL_D2_DS1DS2,
                           Node.VALUE_LABEL_D_DS3, Node.VALUE_LABEL_D2_DS1DS3, Node.VALUE_LABEL_D2_DS2DS3, Node.VALUE_LABEL_D3_DS1DS2DS3]:
            versions = nodetemplate.getValueNumberOfVersions(feField, -1, derivative)
            for v in range(1, versions + 1):
                result, values = feField.getNodeParameters(cache, -1, derivative, v, ncomp)
                if result != ZINC_OK:
                    success = False
                else:
                    newValues = elmult(rotationScale, values)
                    if derivative == Node.VALUE_LABEL_VALUE:
                        newValues = add(newValues, offset)
                    result = feField.setNodeParameters(cache, -1, derivative, v, newValues)
                    if result != ZINC_OK:
                        success = False
        node = nodeIter.next()
    fm.endChange()
    if not success:
        print('zinc.transformCoordinates: failed to get/set some values')
    return success


def getScaffoldNodalParametersToList(field, time=0.0):
    ncomp = field.getNumberOfComponents()
    if (ncomp != 2) and (ncomp != 3):
        print('zinc.transformCoordinates: field has invalid number of components')
        return False
    if field.getCoordinateSystemType() != Field.COORDINATE_SYSTEM_TYPE_RECTANGULAR_CARTESIAN:
        print('zinc.getScaffoldNodalParametersToList: field is not rectangular cartesian')
        return False
    feField = field.castFiniteElement()
    if not feField.isValid():
        print('zinc.getScaffoldNodalParametersToList: field is not finite element field type')
        return False
    success = True
    fm = field.getFieldmodule()
    fm.beginChange()
    cache = fm.createFieldcache()
    cache.setTime(time)
    nodes = fm.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
    nodetemplate = nodes.createNodetemplate()
    nodeIter = nodes.createNodeiterator()
    node = nodeIter.next()
    nodeValueList = list()
    nodeCount = 1
    while node.isValid():
        nodetemplate.defineFieldFromNode(feField, node)
        cache.setNode(node)
        # for derivative in [Node.VALUE_LABEL_VALUE, Node.VALUE_LABEL_D_DS1, Node.VALUE_LABEL_D_DS2, Node.VALUE_LABEL_D2_DS1DS2,
        #                    Node.VALUE_LABEL_D_DS3, Node.VALUE_LABEL_D2_DS1DS3, Node.VALUE_LABEL_D2_DS2DS3, Node.VALUE_LABEL_D3_DS1DS2DS3]:
        versions = nodetemplate.getValueNumberOfVersions(feField, -1, Node.VALUE_LABEL_VALUE)
        for v in range(1, versions + 1):
            result, values = feField.getNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, v, ncomp)
            if result != ZINC_OK:
                success = False
            else:
                nodeValueList.append(values)
        node = nodeIter.next()
        nodeCount += 1
    fm.endChange()
    if not success:
        print('zinc.getScaffoldNodalParametersToList: failed to get/set some values')
    return nodeValueList


def getPointCloudParametersToList(field, time=0.0):
    ncomp = field.getNumberOfComponents()
    if (ncomp != 2) and (ncomp != 3):
        print('zinc.transformCoordinates: field has invalid number of components')
        return False
    if field.getCoordinateSystemType() != Field.COORDINATE_SYSTEM_TYPE_RECTANGULAR_CARTESIAN:
        print('zinc.getScaffoldNodalParametersToList: field is not rectangular cartesian')
        return False
    feField = field.castFiniteElement()
    if not feField.isValid():
        print('zinc.getScaffoldNodalParametersToList: field is not finite element field type')
        return False
    success = True
    fm = field.getFieldmodule()
    fm.beginChange()
    cache = fm.createFieldcache()
    cache.setTime(time)
    nodes = fm.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
    nodetemplate = nodes.createNodetemplate()
    nodeIter = nodes.createNodeiterator()
    node = nodeIter.next()
    nodeValueList = list()
    nodeCount = 1
    while node.isValid():
        nodetemplate.defineFieldFromNode(feField, node)
        cache.setNode(node)
        versions = nodetemplate.getValueNumberOfVersions(feField, -1, Node.VALUE_LABEL_VALUE)
        for v in range(1, versions + 1):
            result, values = feField.getNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, v, ncomp)
            if result != ZINC_OK:
                success = False
            else:
                nodeValueList.append(values)
        node = nodeIter.next()
        nodeCount += 1
    fm.endChange()
    if not success:
        print('zinc.getScaffoldNodalParametersToList: failed to get/set some values')
    return nodeValueList


def setScaffoldNodeParameters(field, newNodeParams, time=0.0):
    ncomp = field.getNumberOfComponents()
    if (ncomp != 2) and (ncomp != 3):
        print('zinc.setScaffoldNodeParameters: field has invalid number of components')
        return False
    if field.getCoordinateSystemType() != Field.COORDINATE_SYSTEM_TYPE_RECTANGULAR_CARTESIAN:
        print('zinc.setScaffoldNodeParameters: field is not rectangular cartesian')
        return False
    feField = field.castFiniteElement()
    if not feField.isValid():
        print('zinc.setScaffoldNodeParameters: field is not finite element field type')
        return False
    success = True
    fm = field.getFieldmodule()
    fm.beginChange()
    cache = fm.createFieldcache()
    cache.setTime(time)
    nodes = fm.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
    nodetemplate = nodes.createNodetemplate()
    nodeIter = nodes.createNodeiterator()
    node = nodeIter.next()
    nodeCount = 0
    while node.isValid():
        nodetemplate.defineFieldFromNode(feField, node)
        cache.setNode(node)
        versions = nodetemplate.getValueNumberOfVersions(feField, -1, Node.VALUE_LABEL_VALUE)
        for v in range(1, versions + 1):
            newValues = newNodeParams[nodeCount]
            result = feField.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, v, newValues)
            if result != ZINC_OK:
                success = False
        node = nodeIter.next()
        nodeCount += 1
    fm.endChange()
    if not success:
        print('zincutils.getScaffoldNodalParametersToList: failed to get/set some values')
    return success


def setDataScale(field, scale):
    if not isinstance(scale, list):
        scale = [scale]*3
    feField = field.castFiniteElement()
    success = True
    fm = field.getFieldmodule()
    fm.beginChange()
    cache = fm.createFieldcache()
    datapoints = fm.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
    nodetemplate = datapoints.createNodetemplate()
    nodeIter = datapoints.createNodeiterator()
    node = nodeIter.next()
    nodeCount = 1
    while node.isValid():
        # nodetemplate.defineFieldFromNode(feField, node)
        cache.setNode(node)
        _, coor = feField.evaluateReal(cache, 3)
        newValues = elmult(coor, scale)
        result = feField.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, newValues)
        if result != ZINC_OK:
            success = False
        node = nodeIter.next()
        nodeCount += 1
    fm.endChange()
    print(nodeCount)
    if not success:
        print('zincutils.setDataScale: failed to get/set some values')
    return success


def swap_axes(source_field, axes=None):
    axis_x = 0
    axis_y = 1
    axis_z = 2
    ncomp = source_field.getNumberOfComponents()
    field = source_field.castFiniteElement()
    if not (field.isValid()):
        print('field must be finite element type')
        return False
    success = True
    fm = field.getFieldmodule()
    fm.beginChange()
    cache = fm.createFieldcache()
    nodes = fm.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
    nodetemplate = nodes.createNodetemplate()
    nodeIter = nodes.createNodeiterator()
    node = nodeIter.next()
    while node.isValid():
        nodetemplate.defineFieldFromNode(field, node)
        cache.setNode(node)
        for derivative in [Node.VALUE_LABEL_VALUE, Node.VALUE_LABEL_D_DS1, Node.VALUE_LABEL_D_DS2,
                           Node.VALUE_LABEL_D2_DS1DS2,
                           Node.VALUE_LABEL_D_DS3, Node.VALUE_LABEL_D2_DS1DS3, Node.VALUE_LABEL_D2_DS2DS3,
                           Node.VALUE_LABEL_D3_DS1DS2DS3]:
            versions = nodetemplate.getValueNumberOfVersions(field, -1, derivative)
            for v in range(1, versions + 1):
                result, values = field.getNodeParameters(cache, -1, derivative, v, ncomp)
                if result != ZINC_OK:
                    success = False
                else:
                    if axes == 'yz':
                        new_values = [values[axis_x], values[axis_z], values[axis_y]]
                        result = field.setNodeParameters(cache, -1, derivative, v, new_values)
                        new_values = [new_values[0], new_values[1], -new_values[2]]
                        result = field.setNodeParameters(cache, -1, derivative, v, new_values)
                    elif axes == 'xz':
                        new_values = [values[axis_z], values[axis_y], values[axis_x]]
                    else:  # 'xy'
                        new_values = [values[axis_y], values[axis_x], values[axis_z]]
                    if result != ZINC_OK:
                        success = False
        node = nodeIter.next()
    fm.endChange()
    if not success:
        print('failed to get/set some values')
    return success
