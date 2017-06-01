# Usage example: calculating the area of a triangle
import pimms

# We make a lazy calculation plan to calculate the area of a triangle; first it calculates the base
# and height, then the area.

# First calc unit: calculate the base and the height
@pimms.calc('base', 'height')
def calc_triangle_base(a, b, c):
    '''
    calc_triangle_base computes the base (x-width) of the triangle a-b-c.
    
    Afferent parameters:
     @ a Must be the (x,y) coordinate of point a in triangle a-b-c.
     @ b Must be the (x,y) coordinate of point b in triangle a-b-c.
     @ c Must be the (x,y) coordinate of point c in triangle a-b-c.

    Efferent values:
     @ base Will be the base, or width, of the triangle a-b-c.
     @ height Will be the height of the triangle a-b-c.
    '''
    print 'Calculating base...'
    xs = [a[0], b[0], c[0]]
    xmin = min(xs)
    xmax = max(xs)
    print 'Calculating height...'
    ys = [a[1], b[1], c[1]]
    ymin = min(ys)
    ymax = max(ys)
    return (xmax - xmin, ymax - ymin)

# Second calc unit: calculate the area
@pimms.calc('area')
def calc_triangle_area(base, height):
    '''
    calc_triangle_are computes the area of a triangle with a given base and height.
    
    Efferent values:
     @ area Will be the area of the triangle with the given base and height.
    '''
    print 'Calculating area...'
    return {'area': base * height * 0.5}
