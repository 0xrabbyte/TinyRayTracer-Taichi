import taichi as ti
from taichi.ui import camera, window
ti.init(arch = ti.gpu)
Pi = 3.141592653
samples = 1500
particles = ti.Vector.field(3, float, samples)

@ti.func
def random_in_unit_sphere():
    #this one is quite fast
    theta = 2.0 * Pi * ti.random()
    phi = ti.acos((2.0 * ti.random()) - 1.0)#this two lines generate two spherical coordinate angles
    r = ti.pow(ti.random(), 1.0/3.0)#you can integrate for each spherical shell in the sphere to prove it
    return ti.Vector([r * ti.sin(phi) * ti.cos(theta), r * ti.sin(phi) * ti.sin(theta), r * ti.cos(phi)])

    #Another possible version, both of them share a idea of generate a random dot on the unit sphere and take a length
    #z = 2.0 * ti.random() - 1.0
    #a = 2.0 * Pi * ti.random()
    #r = ti.sqrt(1.0 - z * z);
    #x = r * ti.cos(a);
    #y = r * ti.sin(a);
    #U = ti.pow(ti.random(), 1.0/3.0)
    #return U * ti.Vector([x, y, z])

    #As slow as the original version, random numbers with a normal distribution is hard for computation
    #U = ti.pow(ti.random(), 1.0/3.0)
    #x, y, z = ti.randn(), ti.randn(), ti.randn()
    #return U * ti.Vector([x, y, z]).normalized()

    #the version in the book, copied from moranzcw's code. the loop will make is π/6 slower
    #p = 2.0 * ti.Vector([ti.random(), ti.random(),
    #                     ti.random()]) - ti.Vector([1, 1, 1])
    #while (p[0] * p[0] + p[1] * p[1] + p[2] * p[2] >= 1.0):
    #    p = 2.0 * ti.Vector(
    #        [ti.random(), ti.random(), ti.random()]) - ti.Vector([1, 1, 1])
    #return p

    #wrong version 1, the naive random on the spherical coordinate system don't have a distribution 
    #def random_in_unit_sphere():
    #r = ti.random()
    #theta = 2.0 * Pi * ti.random()
    #phi = 2.0 * Pi * ti.random()
    #return ti.Vector([r * ti.sin(theta) * ti.cos(phi), r * ti.sin(theta) * ti.sin(phi), r * ti.cos(theta)])

    #wrong version 2
    #return ti.Vector([ti.random(), ti.random(), ti.random()]).normalized()

@ti.kernel
def paint():
    for i in particles:
        particles[i] = random_in_unit_sphere()

gui = ti.ui.Window("A Sphere", (800, 800), vsync=True)
canvas = gui.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()
theta = 0.0
camera.lookat(0, 0, 0)
camera.up(0, 1, 0)
while gui.running:
    paint()
    if gui.is_pressed(ti.ui.LEFT, 'a'): theta -= 5.0
    if gui.is_pressed(ti.ui.RIGHT, 'd'): theta += 5.0
    if (theta > 360.0): theta -= 360.0
    if (theta < 0.0): theta -= 360.0
    x, y = ti.sqrt(18.0) * ti.cos(theta), ti.sqrt(8.0) * ti.sin(theta)
    camera.position(x, 3.0, y)
    scene.set_camera(camera)
    scene.point_light(pos=(0.0, 0.0, 0.0), color = (1, 1, 1))
    scene.particles(particles, radius = 0.02, color = (0.8, 0.3, 0.3))
    canvas.scene(scene)
    gui.show()