import taichi as ti

ti.init(arch=ti.gpu)
Pi = 3.141592653
aspect_ratio = 1.0
image_width = 800
image_height = int(image_width / aspect_ratio)
samples_per_pixel = 4
max_depth = 10

lookfrom = ti.Vector.field(3, float, shape = ())
lookat = ti.Vector.field(3, float, shape = ())
vup = ti.Vector.field(3, float, shape = ())
lookfrom[None] = ti.Vector([0.0, 1.0, -5.0])
lookat[None] = ti.Vector([0.0, 1.0, -1.0])
vup[None] = ti.Vector([0.0, 1.0, 0.0])
vfov = ti.field(float, shape = ())
vfov[None] = 60.0
look_v = ti.Vector([lookfrom[None][0] - lookat[None][0], lookfrom[None][1] - lookat[None][1], lookfrom[None][2] - lookat[None][2]])
dist_to_focus = ti.field(float, shape = ())
dist_to_focus[None] = ti.sqrt(look_v.dot(look_v))
aperture = ti.field(float, shape = ())
aperture[None] = 0.1

@ti.func
def random_in_unit_sphere(): # Here is the optimization
    theta = 2.0 * Pi * ti.random()
    phi = ti.acos((2.0 * ti.random()) - 1.0)
    r = ti.pow(ti.random(), 1.0/3.0)
    return ti.Vector([r * ti.sin(phi) * ti.cos(theta), r * ti.sin(phi) * ti.sin(theta), r * ti.cos(phi)])

@ti.func
def random_in_hemisphere(normal):
    in_unit_sphere = random_in_unit_sphere()
    if (in_unit_sphere.dot(normal) < 0.0):
        in_unit_sphere *= -1
    return in_unit_sphere

@ti.func
def random_in_unit_disk():
    theta = 2.0 * Pi * ti.random()
    return ti.Vector([ti.cos(theta), ti.sin(theta)])

@ti.data_oriented
class ray:
    def __init__(self, origin, direction):
        self.orig = origin
        self.dir = direction
    def origin(self):
        return self.orig
    def direction(self):
        return self.dir
    def at(self, t):
        return self.orig + t * self.dir

@ti.data_oriented
class camera:
    def __init__(self):
        self.viewport_height = ti.field(float, shape = ())
        self.viewport_width = ti.field(float, shape = ())
        self.horizontal = ti.Vector.field(3, float, shape = ())
        self.vertical = ti.Vector.field(3, float, shape = ())
        self.origin = ti.Vector.field(3, float, shape = ())
        self.lower_left_corner = ti.Vector.field(3, float, shape = ())
        self.focal_length = ti.field(float, shape = ())
        self.lens_radius = ti.field(float, shape = ())
        self.reset_view()
    @ti.kernel
    def reset_view(self):
        theta = vfov[None] * Pi / 180.0
        h = ti.tan(theta / 2)
        self.viewport_height[None] = 2.0 * h
        self.viewport_width[None] = aspect_ratio * self.viewport_height[None]

        w = (lookfrom[None] - lookat[None]).normalized()
        u = (vup[None].cross(w)).normalized()
        v = w.cross(u)

        self.focal_length[None] = 1.0
        self.horizontal[None] = dist_to_focus[None] * self.viewport_width[None] * u
        self.vertical[None] = dist_to_focus[None] * self.viewport_height[None] * v
        self.origin[None] = ti.Vector([lookfrom[None][0], lookfrom[None][1], lookfrom[None][2]])
        self.lower_left_corner[None] = self.origin[None] - self.horizontal[None] / 2 - self.vertical[None] / 2 - dist_to_focus[None] * w
        self.lens_radius[None] = aperture[None] / 2.0
    @ti.func
    def get_ray(self, u, v):
        rd = self.lens_radius[None] * random_in_unit_disk()
        offset = u * rd[0] + v * rd[1]
        return ray(self.origin[None] + offset, self.lower_left_corner[None] + u * self.horizontal[None] + v * self.vertical[None] - self.origin[None] - offset)

@ti.func
def set_face_normal(r, outward_normal):
    front_face = (r.direction().dot(outward_normal) < 0)
    normal = outward_normal
    if (not front_face):
        normal = -outward_normal
    return front_face, normal

@ti.func
def reflect(v, n):
    return v - 2.0 * v.dot(n) * n

@ti.func
def refract(uv, n, etai_over_etat):
    cos_theta = min(n.dot(-uv), 1.0)
    r_out_perp = etai_over_etat * (uv + cos_theta * n)
    r_out_parallel = -ti.sqrt(1.0 - r_out_perp.dot(r_out_perp)) * n
    return r_out_perp + r_out_parallel

@ti.func
def reflectance(cosine, ref_idx):
    r0 = (1 - ref_idx) / (1 + ref_idx)
    r0 = r0 * r0
    return r0 + (1 - r0) * pow((1 - cosine), 5)

@ti.data_oriented
class materials:
    def __init__(self, s, a, e):
        self.kind = s
        self.albedo = a
        self.extra = e #Fuzz/Refraction_Rate

@ti.func
def scatter(kind, extra, r, p, normal, ff):
    r_direction = normal
    reached = True
    if (kind == 0): # lambertian
        r_direction = normal + random_in_unit_sphere()
    if (kind == 1): # metal
        r_direction = reflect((r.direction()).normalized(), normal) + extra * random_in_unit_sphere()
        reached = (ray(p, r_direction).direction().dot(normal) > 0)
    if (kind == 2): # dielectric
        refraction_ratio = extra
        if (ff):
            refraction_ratio = 1.0 / extra
        cos_theta = min(normal.dot(-(r.direction()).normalized()), 1.0)
        sin_theta = ti.sqrt(1.0 - cos_theta * cos_theta)
        if (refraction_ratio * sin_theta > 1.0 or reflectance(cos_theta, refraction_ratio) > ti.random()):
            r_direction = reflect((r.direction()).normalized(), normal)
        else:
            r_direction = refract((r.direction()).normalized(), normal, refraction_ratio)
    if (kind == 3): # diffuse_light
        reached = False
    return reached, p, r_direction

@ti.data_oriented
class hittable_list:
    def __init__(self):
        self.objects = []
    def add(self, object):
        self.objects.append(object)
    def clear(self):
        self.objects.clear()
    @ti.func
    def hit(self, r, t_min, t_max):
        closest_so_far = t_max
        t = -1.0
        p = ti.Vector([0.0, 0.0, 0.0])
        front_face = False
        normal = ti.Vector([0.0, 0.0, 0.0])
        mn = 0
        ma = ti.Vector([0.0, 0.0, 0.0])
        me = 0.0
        for i in ti.static(range(len(self.objects))):
            tmp_t, tmp_p, tmp_front_face, tmp_normal, tmp_mn, tmp_ma, tmp_me = (self.objects[i]).hit(r, t_min, closest_so_far)
            if (tmp_t > 0.0):
                closest_so_far = tmp_t
                t, p, front_face, normal, mn, ma, me = tmp_t, tmp_p, tmp_front_face, tmp_normal, tmp_mn, tmp_ma, tmp_me
                maxn = i
        return t, p, front_face, normal, mn, ma, me

@ti.data_oriented
class sphere:
    def __init__(self, cen, r, m):
        self.center = cen
        self.radius = r
        self.material = m
    @ti.func
    def hit(self, r, t_min, t_max):
        oc = r.origin() - self.center
        a = r.direction().dot(r.direction())
        half_b = oc.dot(r.direction())
        c = oc.dot(oc) - self.radius * self.radius
        discriminant = half_b * half_b - a * c
        root = -1.0
        if (discriminant >= 0):
            sqrtd = ti.sqrt(discriminant)
            root = (-half_b - sqrtd) / a
            if (root < t_min or root > t_max):
                root = -1.0
        front_face, normal = False, ti.Vector([0.0, 0.0, 0.0])
        if (root >= 0.0):
            front_face, normal = set_face_normal(r, (r.at(root) - self.center) / self.radius)
        return root, r.at(root), front_face, normal, self.material.kind, self.material.albedo, self.material.extra
                
world = hittable_list()

@ti.func
def ray_color(r, world):
    cnt = 0
    flag = True
    res = ti.Vector([1.0, 1.0, 1.0])
    t, p, front_face, normal, mn, ma, me = world.hit(r, 0.001, float('inf'))
    scattered_o, scattered_d = r.origin(), r.direction()
    if (t > 0):
        while (t > 0 and cnt <= max_depth and flag):
            cnt += 1
            flag, scattered_o, scattered_d = scatter(mn, me, ray(scattered_o, scattered_d), p, normal, front_face)
            res *= ma
            t, p, front_face, normal, mn, ma, me = world.hit(ray(scattered_o, scattered_d), 0.001, float('inf'))     
        if (cnt > max_depth):
            res = ti.Vector([0, 0, 0])
    if (mn != 3):
        unit_direction = scattered_d.normalized()
        t = 0.5 * (unit_direction[1] + 1.0)
        res *= (1.0 - t) * ti.Vector([1.0, 1.0, 1.0]) + t * ti.Vector([0.5, 0.7, 1.0])
    return res

def gen_objects():
    material_ground = materials(0, ti.Vector([0.8, 0.8, 0.8]), 1.0)
    material_left_wall = materials(0, ti.Vector([0.0, 0.6, 0.0]), 1.0)
    material_right_wall = materials(0, ti.Vector([0.6, 0.0, 0.0]), 1.0)
    material_center = materials(0, ti.Vector([0.8, 0.3, 0.3]), 1.0)
    material_left = materials(2, ti.Vector([1.0, 1.0, 1.0]), 1.5)
    material_right = materials(1, ti.Vector([0.6, 0.8, 0.8]), 0.2)
    material_light = materials(3, ti.Vector([10.0, 10.0, 10.0]), 1.0)
    world.add(sphere(ti.Vector([0, -0.2, -1.5]), 0.3, material_center))
    world.add(sphere(ti.Vector([0.7, 0.0, -0.5]), 0.5, material_left))
    world.add(sphere(ti.Vector([-0.8, 0.2, -1.0]), 0.7, material_right))
    world.add(sphere(ti.Vector([0.0, 5.4, -1.0]), 3.0, material_light))
    world.add(sphere(ti.Vector([0.0, -100.5, -1.0]), 100.0, material_ground))
    world.add(sphere(ti.Vector([0.0, 102.5, -1.0]), 100.0, material_ground))
    world.add(sphere(ti.Vector([0.0, 1.0, 101.0]), 100.0, material_ground))
    world.add(sphere(ti.Vector([101.5, 0.0, -1.0]), 100.0, material_left_wall))
    world.add(sphere(ti.Vector([-101.5, 0.0, -1.0]), 100.0, material_right_wall))

    
@ti.kernel
def paint(cnt : int):
    for i, j in pixels:
        col = ti.Vector.zero(float, 3)
        for k in range(samples_per_pixel):
            (u, v) = ((i + ti.random()) / image_width, (j + ti.random()) / image_height)
            r = cam.get_ray(u, v)
            col += ray_color(r, world)
        col /= samples_per_pixel
        radience[i, j] += col
        pixels[i, j] = ti.sqrt(radience[i, j] / ti.cast(cnt, float))

gui = ti.GUI("Tiny Ray Tracer", res = (image_width, image_height))
pixels = ti.Vector.field(3, dtype = float, shape = (image_width, image_height))
radience = ti.Vector.field(3, dtype = float, shape = (image_width, image_height))
gen_objects()
cam = camera()
cnt = 0
is_recording = False
result_dir = "./output"
video_manager = ti.VideoManager(output_dir = result_dir, framerate = 20, automatic_build = False)
while gui.running:
    if gui.get_event(ti.GUI.PRESS):
        if gui.event.key == ti.GUI.LMB:
            x, y = gui.get_cursor_pos()
            lookfrom[None][0] = x * 4.0 - 2.0
            lookfrom[None][1] = y * 2.0 - 0.5
            print("Lookfrom change to ", lookfrom[None])
            look_v = ti.Vector([lookfrom[None][0] - lookat[None][0], lookfrom[None][1] - lookat[None][1], lookfrom[None][2] - lookat[None][2]])
            dist_to_focus[None] = ti.sqrt(look_v.dot(look_v))
            cnt = 0
            radience.fill(0)
            cam.reset_view()
        elif gui.event.key == 's':
            print("Screenshot, spp:", samples_per_pixel * cnt)
            gui.set_image(pixels)
            gui.show("cornellbox.jpg")
        elif gui.event.key == 'j':
            vfov[None] += 5.0
            print("vfov change to ", vfov[None])
            cnt = 0
            radience.fill(0)
            cam.reset_view()
        elif gui.event.key == 'k':
            vfov[None] -= 5.0
            print("vfov change to ", vfov[None])
            cnt = 0
            radience.fill(0)
            cam.reset_view()
        elif gui.event.key == 'r':
            if (is_recording):
                print("Stop Recording")
                video_manager.make_video(gif = True)
            else:
                print("Start recording")
            is_recording = not is_recording
            cnt = 0
            radience.fill(0)
            cam.reset_view()
        elif gui.event.key == ti.GUI.ESCAPE:
            exit()
    cnt += 1
    paint(cnt)
    gui.set_image(pixels)
    gui.show()
    if (is_recording):
        video_manager.write_frame(pixels)