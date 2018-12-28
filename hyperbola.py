from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc
from matplotlib.transforms import IdentityTransform
import numpy as np

min_val = -1.5  # min value
max_val = 1.5  # max value
step = 0.125
height = 70
a = 1
diameter = 115


class AngleMarker(Arc):
    def __init__(self, xy, size, vec1, vec2, ax=None, **kwargs):
        self._xydata = xy  # in data coordinates
        self.ax = ax or plt.gca()
        self.vec1 = vec1  # tuple or array of coordinates, relative to xy
        self.vec2 = vec2  # tuple or array of coordinates, relative to xy

        super().__init__(self._xydata, size, size, angle=0.0,
                         theta1=self.theta1, theta2=self.theta2, **kwargs)

        self.set_transform(IdentityTransform())
        self.ax.add_patch(self)

    def get_center_pixels(self):
        """ return center in pixel coordinates """
        return self.ax.transData.transform(self._xydata)

    def set_center(self, xy):
        """ set center in data coordinates """
        self._xydata = xy

    _center = property(get_center_pixels, set_center)

    def get_theta(self, vec):
        vec_in_pixels = self.ax.transData.transform(vec) - self._center
        return np.rad2deg(np.arctan2(vec_in_pixels[1], vec_in_pixels[0]))

    def get_theta1(self):
        return self.get_theta(self.vec1)

    def get_theta2(self):
        return self.get_theta(self.vec2)

    def set_theta(self, angle):
        pass

    theta1 = property(get_theta1, set_theta)
    theta2 = property(get_theta2, set_theta)


class Point2D:
    x, y = (float(0), float(0))

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def text(self, pos='top'):
        if pos == 'top':
            plt.text(self.x + step / 5, self.y + step / 5, str((round(self.x, 4), round(self.y, 4))), fontsize=12,
                     verticalalignment='bottom')
        elif pos == 'bottom':
            plt.text(self.x + step / 5, self.y - step / 5, str((round(self.x, 4), round(self.y, 4))), fontsize=12,
                     verticalalignment='top')
        return None

    def dist(self, pt):
        return np.sqrt((self.x - pt.x) ** 2 + (self.y - pt.y) ** 2)

    def show(self, color='black'):
        plt.plot(self.x, self.y, 'o', color=color)
        return None

    def show_seg(self, pt, color='black', style='o-'):
        plt.plot([self.x, pt.x], [self.y, pt.y], style, color=color)
        return None

    def to_tup(self):
        return self.x, self.y

    def cx(self, new_x):
        return Point2D(new_x, self.y)

    def cy(self, new_y):
        return Point2D(self.x, new_y)


# draw segment with distance in mm
def distance_seg(point1, point2, scale, pos='right_y', color='black', text_pos='top'):
    x1 = point1.x
    x2 = point2.x
    y1 = point1.y
    y2 = point2.y
    if pos == 'right_y':
        plt.plot([max_val-step, max_val-step], [y1, y2], 'o-', color=color)
        plt.plot([max_val-step, max_val-step*2], [y1, y1], '-', color=color)
        plt.plot([max_val-step, max_val-step*2], [y2, y2], '-', color=color)

        d = abs(y1 - y2)
        plt.text(max_val-step*1.5, d/2 + min(y1, y2), str(round(d*scale, 2))+' mm.', fontsize=18,
                 color=color, verticalalignment='center', horizontalalignment='center', rotation='vertical')
    if pos == 'left_y':
        plt.plot([min_val + step, min_val + step], [y1, y2], 'o-', color=color)
        plt.plot([min_val + step, min_val + step * 2], [y1, y1], '-', color=color)
        plt.plot([min_val + step, min_val + step * 2], [y2, y2], '-', color=color)

        d = abs(y1 - y2)
        plt.text(min_val + step * 1.5, d / 2 + min(y1, y2), str(round(d * scale, 2)) + ' mm.', fontsize=18,
                 color=color, verticalalignment='center', horizontalalignment='center', rotation='vertical')
    if pos == 'x':
        plt.plot([x1, x2], [0, 0], 'o-', color=color)
        plt.plot([x1, x1], [0, step], '-', color=color)
        plt.plot([x2, x2], [0, step], '-', color=color)

        d = abs(x1 - x2)
        if text_pos == 'top':
            plt.text((x1 + x2)/2, step * 0.5, str(round(d*scale, 2))+' mm.', fontsize=18,
                     color=color, verticalalignment='center', horizontalalignment='center')
        elif text_pos == 'bottom':
            plt.text((x1 + x2) / 2, -step * 0.5, str(round(d * scale, 2)) + ' mm.', fontsize=18,
                     color=color, verticalalignment='center', horizontalalignment='center')


fig, ax = plt.subplots(figsize=(15.5, 15))

x = np.linspace(min_val-10, max_val+10, 500000)


# get all of the intersections of two graphs
def get_intersection(y1, y2):
    global x
    index = np.argwhere(np.diff(np.sign(y1 - y2))).flatten()
    return [Point2D(x[i], y1[i]) for i in index]


# create a line from two points
def get_line(pt1, pt2):
    global x
    cf = np.polyfit((pt1.x, pt2.x), (pt1.y, pt2.y), 1)
    p = np.poly1d(cf)
    return p(x)


# hyperbola function
b = 1  # i don't see any point in changing b, as for now
c = np.sqrt(a**2 + b**2)
f1 = Point2D(0, c)  # first focal point
f2 = Point2D(0, -c)  # second focal point
y = np.sqrt(x**2/a**2+1)*b  # our hyperbola graph

# calculating scale
scale = height/f1.y/2

# second slice lines
# left part
# reflection
tptl = Point2D(-diameter/scale, f2.y-65/scale)  # target point (left)
y_left_ref = get_line(f1, tptl)
# point
ptl = get_intersection(y, y_left_ref)[0]

# right part
# reflection
tptr = Point2D(diameter/scale, f2.y-65/scale)  # target point (right)
y_right_ref = get_line(f1, tptr)
# point
ptr = get_intersection(y, y_right_ref)[-1]

# setting up appearance of the plot - start
plt.axis('equal')

plt.yticks(np.arange(min_val, max_val+1, step=step))
plt.xticks(np.arange(min_val, max_val+1, step=step))

ax.set_xlim([min_val, max_val])
ax.set_ylim([min_val, max_val])

plt.axhline(0, color='black')
plt.axvline(0, color='black')
# setting up appearance of the plot - end

# hyperbola itself
plt.plot(x, y, '#f15a22')

# plotting left part
plt.plot(x, y_left_ref, color=(0.3, 0.3, 0.3))
f2.show_seg(ptl, style='-', color='green')

# plotting right part
plt.plot(x, y_right_ref, color=(0.3, 0.3, 0.3))
f2.show_seg(ptr, style='-', color='green')

# bottom slice
plt.axhline(ptl.y, linestyle='--', color='purple')
ptr.show_seg(ptl, style='-', color='purple')

# top slice
plt.axhline(f1.y, linestyle='--', color='purple')
tsp = get_intersection(0*x+f1.y, y)
tsp[0].show_seg(tsp[1], style='-', color='purple')

# calculating and plotting angle
f2.show_seg(tsp[0], style='-', color='blue')
f2.show_seg(tsp[1], style='-', color='green')
f2.show_seg(f1, style='-', color='blue')
tsp[0].show_seg(tsp[0].cx(min_val-10), style='-', color='green')
tsp[1].show_seg(tsp[0].cx(max_val+10), style='-', color='green')
alpha = f1.y*2/f2.dist(tsp[0])
AngleMarker(f2.to_tup(), 600, f1.to_tup(), tsp[0].to_tup(), ax=ax, color='blue')

# vertex point
vertex = Point2D(0, b)

# distances in mm
distance_seg(f1, f2, scale, 'right_y', '#913d88')
distance_seg(f1, vertex, scale, 'right_y', '#4d13d1')
distance_seg(tsp[1], tsp[0], scale, 'x', '#913d88', 'bottom')
distance_seg(ptl, ptr, scale, 'x', '#4d13d1')
distance_seg(ptl, f1, scale, 'left_y', '#4d13d1')

# text with coordinates nearby points
f1.text()
f2.text()
ptl.text()
ptr.text()
ptl.show('purple')
ptr.show('purple')
for i in tsp:
    i.show('purple')
    i.text()
vertex.text('bottom')
vertex.show('#f15a22')

# top focal point
f1.show('red')

# bottom focal point
f2.show('red')

# legend
patches = []
patches.append(mpatches.Patch(color='#f15a22', label='hyperbola (mirror)'))
patches.append(mpatches.Patch(color='red', label='focal points'))
patches.append(mpatches.Patch(color='purple', label='slices'))
patches.append(mpatches.Patch(color='blue', label='alpha'))
patches.append(mpatches.Patch(color='green', label="rays from camera's focal point"))
plt.legend(handles=patches, loc='lower left')

# textbox
textstr = '\n'.join((
    r'$\mathrm{height}=%.0f$ mm.' % (height, ),
    r'$\mathrm{diameter}=%.0f$ mm.' % (diameter, ),
    r'$\mathrm{a}=%.2f$' % (a, ),
    r'$\mathrm{b}=%.2f$' % (b, ),
    r'$\mathrm{scale}=1:%.2f$' % (scale, ),
    r'$\alpha=%.2f\degree$' % (float(np.degrees(np.arccos(alpha))), )))
props = dict(boxstyle='round', facecolor=(0.97, 0.97, 0.97), alpha=0.5)
ax.text(0.01, 0.03 + len(patches)*0.012, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='bottom', bbox=props)

# setting up appearance of the plot - start
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('gray')
ax.spines['bottom'].set_color('gray')

plt.grid(True, linestyle='--')
# setting up appearance of the plot - end

# show the plot
plt.show()
