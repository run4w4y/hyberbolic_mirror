from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

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
    index = np.argwhere(np.diff(np.sign(y1 - y2))).flatten()
    r = []
    for i in index:
        tx = x[i]
        ty = y1[i]
        r.append((tx, ty))
    return r


# add text with coordinates nearby point
def text_point(pt):
    plt.text(pt[0] + step / 5, pt[1] + step / 5, str((round(pt[0], 4), round(pt[1], 4))), fontsize=12)
    return None


# distance between two points
def dist(pt1, pt2):
    return np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)


# create a line from two points
def get_line(pt1, pt2):
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
plt.plot(x, y, 'green')

# plotting left part
plt.plot(x, y_left_ref, 'blue')
plt.plot([0, ptl[0]], [f2, ptl[1]], 'b-')

# plotting right part
plt.plot(x, y_right_ref, 'blue')
plt.plot([0, ptr[0]], [f2, ptr[1]], 'b-')

# bottom slice
plt.axhline(ptl[1], linestyle='--', color='purple')
plt.plot([ptl[0], ptr[0]], [ptl[1], ptr[1]], '-', color='purple')

# top slice
plt.axhline(f1, linestyle='--', color='purple')
tsp = get_intersection(0*x+f1, y)
plt.plot([tsp[0][0], tsp[1][0]], [tsp[0][1], tsp[1][1]], '-', color='purple')

# calculating and plotting angle

for i in tsp:
    plt.plot([i[0], 0], [i[1], f2], '-', color='blue')
alpha = f1*2/dist(tsp[0], (0, f2))

# top focal point
plt.plot(0, f1, 'ro')

# bottom focal point
plt.plot(0, f2, 'ro')

# text with coordinates nearby points
text_point((0, f1))
text_point((0, f2))
text_point(ptl)
text_point(ptr)
plt.plot(ptl[0], ptl[1], 'bo')
plt.plot(ptr[0], ptr[1], 'bo')
for i in tsp:
    plt.plot(i[0], i[1], 'o', color='purple')
    text_point(i)

# legend
patches = []
patches.append(mpatches.Patch(color='green', label='hyperbola (mirror)'))
patches.append(mpatches.Patch(color='red', label='focal points'))
patches.append(mpatches.Patch(color='purple', label='slices'))
plt.legend(handles=patches, loc='lower left')

# textbox
textstr = '\n'.join((
    r'$\mathrm{height}=%.0f$' % (height, ),
    r'$\mathrm{diameter}=%.0f$' % (diameter, ),
    r'$\mathrm{a}=%.2f$' % (a, ),
    r'$\mathrm{scale}=1:%.2f$' % (scale, ),
    r'$\alpha=%.2f$deg' % (float(np.degrees(np.arccos(alpha))), )))
props = dict(boxstyle='round', facecolor=(0.97, 0.97, 0.97), alpha=0.5)
ax.text(0.01, 0.06, textstr, transform=ax.transAxes, fontsize=14,
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
