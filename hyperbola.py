from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Arc
from matplotlib.transforms import IdentityTransform
import numpy as np


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


fig, ax = plt.subplots(figsize=(15.5, 15))

min_val = -1.5  # min value
max_val = 1.5  # max value

x = np.linspace(min_val-10, max_val+10, 500000)

step = 0.125
height = 70
a = 1
diameter = 115


# get all of the intersections of two graphs
def get_intersection(y1, y2):
    global x
    index = np.argwhere(np.diff(np.sign(y1 - y2))).flatten()
    r = []
    for i in index:
        tx = x[i]
        ty = y1[i]
        r.append((tx, ty))
    return r


# add text with coordinates nearby point
def text_point(pt, pos='top'):
    if pos == 'top':
        plt.text(pt[0] + step / 5, pt[1] + step / 5, str((round(pt[0], 4), round(pt[1], 4))), fontsize=12,
                 verticalalignment='bottom')
    elif pos == 'bottom':
        plt.text(pt[0] + step / 5, pt[1] - step / 5, str((round(pt[0], 4), round(pt[1], 4))), fontsize=12,
                 verticalalignment='top')
    return None


# distance between two points
def dist(pt1, pt2):
    return np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)


# create a line from two points
def get_line(pt1, pt2):
    global x
    cf = np.polyfit((pt1[0], pt2[0]), (pt1[1], pt2[1]), 1)
    p = np.poly1d(cf)
    return p(x)


# hyperbola function
b = 1  # i don't see any point in changing b, as for now
c = np.sqrt(a**2 + b**2)
f1 = c  # first focal point
f2 = -c  # second focal point
y = np.sqrt(x**2/a**2+1)*b  # our hyperbola graph

# calculating scale
scale = height/f1/2

# second slice lines
# left part
# reflection
tptl = (-diameter/scale, f2-65/scale)  # target point (left)
y_left_ref = get_line((0, f1), tptl)
# point
ptl = get_intersection(y, y_left_ref)[0]

# right part
# reflection
tptr = (diameter/scale, f2-65/scale)  # target point (right)
y_right_ref = get_line((0, f1), tptr)
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
plt.plot([0, ptl[0]], [f2, ptl[1]], '-', color='green')

# plotting right part
plt.plot(x, y_right_ref, color=(0.3, 0.3, 0.3))
plt.plot([0, ptr[0]], [f2, ptr[1]], '-', color='green')

# bottom slice
plt.axhline(ptl[1], linestyle='--', color='purple')
plt.plot([ptl[0], ptr[0]], [ptl[1], ptr[1]], '-', color='purple')

# top slice
plt.axhline(f1, linestyle='--', color='purple')
tsp = get_intersection(0*x+f1, y)
plt.plot([tsp[0][0], tsp[1][0]], [tsp[0][1], tsp[1][1]], '-', color='purple')

# calculating and plotting angle
plt.plot([tsp[0][0], 0], [tsp[0][1], f2], '-', color='blue')
plt.plot([tsp[1][0], 0], [tsp[1][1], f2], '-', color='green')
plt.plot([0, 0], [f1, f2], '-', color='blue')
plt.plot([tsp[0][0], min_val - 10], [tsp[0][1], tsp[0][1]], '-', color='green')
plt.plot([tsp[1][0], max_val + 10], [tsp[1][1], tsp[0][1]], '-', color='green')
alpha = f1*2/dist(tsp[0], (0, f2))
AngleMarker((0, f2), 600, (0, f1), tsp[0], ax=ax, color='blue')

# vertex point
vertex = (0, b)

# height segment
plt.plot([max_val-step, max_val-step], [f1, f2], 'o-', color='#913d88')
plt.plot([max_val-step, max_val-step*2], [f1, f1], '-', color='#913d88')
plt.plot([max_val-step, max_val-step*2], [f2, f2], '-', color='#913d88')

# height text
plt.text(max_val-step*1.5, 0, 'height', fontsize=18, color='#913d88', verticalalignment='center',
         horizontalalignment='center', rotation='vertical')

# lens' height segment
plt.plot([max_val-step, max_val-step], [f1, vertex[1]], 'o-', color='#4d13d1')
plt.plot([max_val-step, max_val-step*2], [f1, f1], '-', color='#4d13d1')
plt.plot([max_val-step, max_val-step*2], [vertex[1], vertex[1]], '-', color='#4d13d1')

# lens' height text
lensl = f1 - vertex[1]
plt.text(max_val-step*1.5, lensl/2 + vertex[1], str(round(lensl*scale, 2))+' mm.', fontsize=18, color='#4d13d1',
         verticalalignment='center', horizontalalignment='center', rotation='vertical')

# top slice's width segment
plt.plot([tsp[0][0], tsp[1][0]], [0, 0], 'o-', color='#913d88')
plt.plot([tsp[0][0], tsp[0][0]], [0, step], '-', color='#913d88')
plt.plot([tsp[1][0], tsp[1][0]], [0, step], '-', color='#913d88')

# top slice's width text
plt.text(0, -step*0.5, str(round(abs(tsp[0][0] - tsp[1][0])*scale, 2))+' mm.', fontsize=18, color='#663399',
         verticalalignment='center', horizontalalignment='center')

# bottom slice's width segment
plt.plot([ptl[0], ptr[0]], [0, 0], 'o-', color='#4d13d1')
plt.plot([ptl[0], ptl[0]], [0, step], '-', color='#4d13d1')
plt.plot([ptr[0], ptr[0]], [0, step], '-', color='#4d13d1')

# bottom slice's width text
plt.text(0, step*0.5, str(round(abs(ptl[0] - ptr[0])*scale, 2))+' mm.', fontsize=18, color='#4d13d1',
         verticalalignment='center', horizontalalignment='center')

# text with coordinates nearby points
text_point((0, f1))
text_point((0, f2))
text_point(ptl)
text_point(ptr)
plt.plot(ptl[0], ptl[1], 'o', color='purple')
plt.plot(ptr[0], ptr[1], 'o', color='purple')
for i in tsp:
    plt.plot(i[0], i[1], 'o', color='purple')
    text_point(i)
text_point(vertex, pos='bottom')
plt.plot(vertex[0], vertex[1], 'o', color='#f15a22')

# top focal point
plt.plot(0, f1, 'ro')

# bottom focal point
plt.plot(0, f2, 'ro')

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
