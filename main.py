import requests
import numpy as np
import json
from io import BytesIO
from PIL import Image
import base64
import struct
import math
from pye57 import E57

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

lat = "40.457375"
lon = "-80.009353"

pano = {"pano_id": None, "depthmap": None}
image_counter = 0
total_images = 0
pano_tiles = []
depthmap_image = None
point_cloud = []
pano_image = None

gl_scale = 1.0
start_drag_x = 0
start_drag_y = 0
rotation_x = 0
rotation_y = 0


def construct_depth_map():
    depth_map_compressed = []
    for i in range(0, len(pano["depthmap"]), 4):
        try:
            depth_map_compressed += base64.b64decode(
                s=pano["depthmap"][i : i + 4], altchars=b"-_", validate=True
            )
        except:
            depth_map_compressed += b"\x01\x01\x01"

    depth_map_compressed = bytearray(depth_map_compressed)

    headersize = int(depth_map_compressed[0])
    numberofplanes = int(depth_map_compressed[1] | (depth_map_compressed[2] << 8))
    width = int(depth_map_compressed[3] | (depth_map_compressed[4] << 8))
    height = int(depth_map_compressed[5] | (depth_map_compressed[6] << 8))
    offset = int(depth_map_compressed[7])

    if headersize != 8 or offset != 8:
        print("Unexpected depth map header ")
        return False

    depthmap_indices = depth_map_compressed[offset : offset + height * width]
    temp_array = depth_map_compressed[
        offset + height * width : offset + height * width + numberofplanes * 16
    ]

    float_array = []
    for i in range(0, numberofplanes * 16, 4):
        float_array.append(struct.unpack("f", temp_array[i : i + 4])[0])

    depthmap_planes = []
    for i in range(0, len(float_array), 4):
        try:
            depthmap_planes.append(
                {
                    "x": float_array[i],
                    "y": float_array[i + 1],
                    "z": float_array[i + 2],
                    "d": float_array[i + 3],
                }
            )
        except:
            pass

    depthmap = [0] * (width * height)
    for y in range(0, height, 1):
        for x in range(0, width, 1):
            xnormalize = (width - x - 1.0) / (width - 1.0)
            ynormalize = (height - y - 1.0) / (height - 1.0)

            theta = xnormalize * 2 * math.pi + math.pi / 2.0
            phi = ynormalize * math.pi

            v = []
            v.append(math.sin(phi) * math.cos(theta))
            v.append(math.sin(phi) * math.sin(theta))
            v.append(math.cos(phi))

            planeIdx = depthmap_indices[y * width + x]
            if planeIdx > 0:
                plane = depthmap_planes[planeIdx]
                t = abs(
                    plane["d"]
                    / (v[0] * plane["x"] + v[1] * plane["y"] + v[2] * plane["z"])
                )
                depthmap[y * width + width - x - 1] = min(255, int(t / 100.0 * 255.0))
            else:
                depthmap[y * width + width - x - 1] = 0

    global depthmap_image
    depthmap_image = Image.fromarray(np.array(depthmap).reshape(height, width))


def construct_point_cloud():
    global point_cloud
    point_cloud = []
    width = depthmap_image.width
    height = depthmap_image.height

    for y in range(0, height, 1):
        for x in range(0, width, 1):
            xnormalize = (width - x - 1.0) / (width - 1.0)
            ynormalize = (height - y - 1.0) / (height - 1.0)

            theta = xnormalize * 2 * math.pi + math.pi / 2
            phi = ynormalize * math.pi

            depth = depthmap_image.getpixel((x, y))
            if depth > 100:
                depth = 100

            xx = math.sin(phi) * math.cos(theta)
            yy = math.sin(phi) * math.sin(theta)
            zz = math.cos(phi)

            if depth != 0:
                pos = (xx * depth, yy * depth, zz * depth)
            else:
                pos = (xx * 100.0, yy * 100.0, zz * 100.0)

            global pano_image
            r = pano_image.getpixel((x, y))[0] / 255.0
            g = pano_image.getpixel((x, y))[1] / 255.0
            b = pano_image.getpixel((x, y))[2] / 255.0
            col = (r, g, b)

            point_cloud.append({"pos": pos, "color": col})

    point_cloud_e57 = {
        "cartesianX": np.float32([point["pos"][0] for point in point_cloud]),
        "cartesianY": np.float32([point["pos"][1] for point in point_cloud]),
        "cartesianZ": np.float32([point["pos"][2] for point in point_cloud]),
        "colorRed": np.float32([point["color"][0] for point in point_cloud]),
        "colorGreen": np.float32([point["color"][1] for point in point_cloud]),
        "colorBlue": np.float32([point["color"][2] for point in point_cloud]),
    }
    with E57("point_cloud.e57", mode="w") as e:
        e.write_scan_raw(point_cloud_e57)


def construct_pano_image():
    zoom = 1
    w = pow(2, zoom)
    h = pow(2, zoom - 1)
    image_counter = 0
    total_images = w * h

    for y in range(h):
        for x in range(w):
            url = (
                "https://maps.google.com/cbk?output=tile&panoid="
                + pano["pano_id"]
                + "&zoom="
                + str(zoom)
                + "&x="
                + str(x)
                + "&y="
                + str(y)
            )
            response = requests.get(url)

            if response.status_code == 200:
                tile = {}
                tile["x"] = x
                tile["y"] = y
                tile["image"] = Image.open(BytesIO(response.content))

                pano_tiles.append(tile)
                image_counter += 1

                if image_counter == total_images:
                    image_size = 512
                    w = image_size * pow(2, zoom)
                    h = image_size * pow(2, zoom - 1)

                    global pano_image
                    pano_image = Image.new(
                        "RGB",
                        (
                            pano_tiles[0]["image"].width * len(pano_tiles),
                            pano_tiles[0]["image"].height,
                        ),
                    )
                    for item in pano_tiles:
                        pano_image.paste(
                            item["image"], (item["image"].width * item["x"], 0)
                        )

                    pano_image = pano_image.resize((512, 256)).transpose(
                        method=Image.Transpose.FLIP_LEFT_RIGHT
                    )

                    construct_depth_map()

                    construct_point_cloud()


def get_street_view_data():
    requestURL = (
        "https://maps.googleapis.com/maps/api/js/GeoPhotoService.SingleImageSearch?pb=!1m5!1sapiv3!5sUS!11m2!1m1!1b0!2m4!1m2!3d"
        + lat
        + "!4d"
        + lon
        + "!2d50!3m10!2m2!1sen!2sGB!9m1!1e2!11m4!1m3!1e2!2b1!3e2!4m10!1e1!1e2!1e3!1e4!1e8!1e6!5m1!1e2!6m1!1e2&callback=_xdc_._v2mub5"
    )
    response = requests.get(requestURL)

    if response.status_code == 200:
        responseDataStr = response.content.decode()
        responseDataStr = responseDataStr[
            responseDataStr.index("(") + 1 : responseDataStr.rindex(")") - 1
        ]
        responseDataJson = json.loads(responseDataStr)
        if len(responseDataJson) == 1:
            print("Google Street View is not available")
            exit()

        pano["pano_id"] = responseDataJson[1][1][1]

        requestURL = (
            "https://www.google.com/maps/photometa/v1?authuser=0&output=xml&hl=en&gl=uk&pb=!1m4!1smaps_sv.tactile!11m2!2m1!1b1!2m2!1sen!2suk!3m3!1m2!1e2!2s"
            + pano["pano_id"]
            + "!4m57!1e1!1e2!1e3!1e4!1e5!1e6!1e8!1e12!2m1!1e1!4m1!1i48!5m1!1e1!5m1!1e2!6m1!1e1!6m1!1e2!9m36!1m3!1e2!2b1!3e2!1m3!1e2!2b0!3e3!1m3!1e3!2b1!3e2!1m3!1e3!2b0!3e3!1m3!1e8!2b0!3e3!1m3!1e1!2b0!3e3!1m3!1e4!2b0!3e3!1m3!1e10!2b1!3e2!1m3!1e10!2b0!3e3"
        )
        response = requests.get(requestURL)

        if response.status_code == 200:
            responseDataStr = response.content.decode()
            responseDataJson = json.loads(responseDataStr[4:])
            pano["depthmap"] = responseDataJson[1][0][5][0][5][1][2]

            construct_pano_image()


def draw_point_cloud():
    glBegin(GL_POINTS)
    for point in point_cloud:
        glColor3fv(point["color"])
        glVertex3fv(point["pos"])
    glEnd()


def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glEnable(GL_TEXTURE_2D)

    glViewport(0, 600, 512, 256)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-1, 1, -1, 1, -1, 1.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glBindTexture(GL_TEXTURE_2D, tex_pano_id)
    glBegin(GL_QUADS)
    glTexCoord2f(1.0, 1.0)
    glVertex3f(-1.0, -1.0, 0.0)
    glTexCoord2f(0.0, 1.0)
    glVertex3f(1.0, -1.0, 0.0)
    glTexCoord2f(0.0, 0.0)
    glVertex3f(1.0, 1.0, 0.0)
    glTexCoord2f(1.0, 0.0)
    glVertex3f(-1.0, 1.0, 0.0)
    glEnd()

    glViewport(512, 600, 512, 256)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-1, 1, -1, 1, -1, 1.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    glBindTexture(GL_TEXTURE_2D, tex_depth_id)
    glBegin(GL_QUADS)
    glTexCoord2f(1.0, 1.0)
    glVertex3f(-1.0, -1.0, 0.0)
    glTexCoord2f(0.0, 1.0)
    glVertex3f(1.0, -1.0, 0.0)
    glTexCoord2f(0.0, 0.0)
    glVertex3f(1.0, 1.0, 0.0)
    glTexCoord2f(1.0, 0.0)
    glVertex3f(-1.0, 1.0, 0.0)
    glEnd()

    glDisable(GL_TEXTURE_2D)

    glViewport(0, 0, 1024, 600)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(80, (1024 / 600), 0.1, 1000.0)
    gluLookAt(300, 0, 0, 0, 0, 0, 0, 0, -1)
    glMatrixMode(GL_MODELVIEW)

    glLoadIdentity()
    glRotate(-rotation_y, 0.0, 1.0, 0.0)
    glRotate(-rotation_x, 0.0, 0.0, 1.0)
    glScalef(gl_scale, gl_scale, gl_scale)

    glCallList(list_id)

    glutSwapBuffers()


def mouse_wheel(button, dir, x, y):
    global gl_scale
    if dir > 0:
        gl_scale += 0.1
    elif dir < 0:
        gl_scale -= 0.1
    if gl_scale < 0.1:
        gl_scale = 0.1
    glutPostRedisplay()


def mouse(button, state, x, y):
    global start_drag_x, start_drag_y
    if button == GLUT_LEFT_BUTTON and state == GLUT_DOWN:
        start_drag_x, start_drag_y = x, y
    glutPostRedisplay()


def motion(x, y):
    global rotation_x, rotation_y, start_drag_x, start_drag_y
    rotation_y += (y - start_drag_y) * 0.3
    rotation_x += (x - start_drag_x) * 0.3
    start_drag_x, start_drag_y = x, y
    glutPostRedisplay()


def gl_init():
    global list_id
    global tex_pano_id
    global tex_depth_id
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_TEXTURE_2D)
    glClearColor(0.1, 0.1, 0.1, 1)
    glClearDepth(1.0)

    list_id = glGenLists(1)
    glNewList(list_id, GL_COMPILE)
    draw_point_cloud()
    glEndList()

    tex_pano_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_pano_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGB,
        pano_image.width,
        pano_image.height,
        0,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        pano_image.tobytes(),
    )

    tex_depth_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_depth_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGB,
        depthmap_image.width,
        depthmap_image.height,
        0,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        depthmap_image.convert("RGB").tobytes(),
    )

    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)


def main():
    get_street_view_data()

    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH)
    glutInitWindowSize(1024, 856)
    glutInitWindowPosition(100, 100)
    glutCreateWindow("Google Street View - Point Cloud")

    gl_init()

    glutDisplayFunc(display)
    glutMouseFunc(mouse)
    glutMotionFunc(motion)
    glutMouseWheelFunc(mouse_wheel)
    glutMainLoop()


main()
