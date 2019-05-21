
def createFiniteElementField(region, field_name='coordinates'):
    field_module = region.getFieldmodule()
    field_module.beginChange()

    finite_element_field = field_module.createFieldFiniteElement(3)

    finite_element_field.setName(field_name)

    finite_element_field.setManaged(True)
    finite_element_field.setTypeCoordinate(True)
    field_module.endChange()

    return finite_element_field


def getMesh(region):
    fm = region.getFieldmodule()
    for dimension in range(3, 0, -1):
        mesh = fm.findMeshByDimension(dimension)
        if mesh.getSize() > 0:
            return mesh
    raise ValueError('Model contains no mesh')


def getModelCoordinateField(region, modelReferenceCoordinateField):
    mesh = getMesh(region)
    element = mesh.createElementiterator().next()
    if not element.isValid():
        raise ValueError('Model contains no elements')
    fm = region.getFieldmodule()
    cache = fm.createFieldcache()
    cache.setElement(element)
    fieldIter = fm.createFielditerator()
    field = fieldIter.next()
    while field.isValid():
        if field.isTypeCoordinate() and (field.getNumberOfComponents() <= 3) and\
                ((modelReferenceCoordinateField is None) or
                 (field != modelReferenceCoordinateField)):
            if field.isDefinedAtLocation(cache):
                return field
        field = fieldIter.next()
    raise ValueError('Could not determine model coordinate field')


