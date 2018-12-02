import dominate
import matplotlib.pyplot as plt
from matplotlib import animation
from dominate import tags
from scipy.misc import imsave as sp_imsave
from skimage import img_as_int
from bs4 import BeautifulSoup
import os
from io import BytesIO
from base64 import b64encode, b64decode
import datetime
import imageio
import matplotlib.image as mpimg


def format_dict(d):
    s = ['']

    def helper(d, s, depth=0):
        for k, v in sorted(d.items(), key=lambda x: x[0]):
            if isinstance(v, dict):
                s[0] += ("  ")*depth + ("%s: {" % k) + ',\n'
                helper(v, s, depth+1)
                s[0] += ("  ")*depth + ("}") + ',\n'
            else:
                s[0] += ("  ")*depth + "%s: %s" % (k, v) + ',\n'

    helper(d, s)
    return s[0]


class HTMLReport(object):
    def __init__(self, path, images_per_row=1000, default_image_width=400):
        self.path = path
        title = datetime.datetime.today().strftime(
            "Report %Y-%m-%d_%H-%M-%S_{}".format(os.uname()[1])
        )
        self.doc = dominate.document(title=title)
        self.images_per_row = images_per_row
        self.default_image_width = default_image_width
        self.t = None
        self.row_image_count = 0

    def add_header(self, str):
        with self.doc:
            tags.h3(str, style='word-wrap: break-word; white-space: pre-wrap;')
        self.t = None
        self.row_image_count = 0

    def add_text(self, str):
        with self.doc:
            tags.p(str, style='word-wrap: break-word; white-space: pre-wrap;')
        self.t = None
        self.row_image_count = 0

    def _add_table(self, border=1):
        self.row_image_count = 0
        self.t = tags.table(border=border, style="table-layout: fixed;")
        self.doc.add(self.t)

    def _encode_image(self, img_arr):
        """Save the image array as PNG and then encode with base64 for embedding"""
        img_arr = img_as_int(img_arr)
        sio = BytesIO()
        sp_imsave(sio, img_arr, 'png')
        encoded = b64encode(sio.getvalue()).decode()
        sio.close()
        return encoded

    def add_image(self, im, txt='', width=None, font_pct=100):
        if width is None:
            width = self.default_image_width
        if self.t is None or self.row_image_count >= self.images_per_row:
            self._add_table()
        with self.t:
            # with tr():
            # with td(style="word-wrap: break-word;", halign="center", valign="top"):
            with tags.td(halign="center", valign="top"):
                with tags.p():
                    tags.img(
                        style="width:%dpx" % width,
                        src=r'data:image/png;base64,' + self._encode_image(im)
                    )
                    tags.br()
                    tags.p(
                        txt,
                        style='width:{}px; word-wrap: break-word; white-space: pre-wrap; font-size: {}%;'.format(
                            width,
                            font_pct
                        )
                    )
        self.row_image_count += 1

    def new_row(self):
        self.save()
        self.t = None
        self.row_image_count = 0

    def add_images(self, ims, txts, width=256):
        for im, txt in zip(ims, txts):
            self.add_image(im, txt, width)

    def save(self):
        with open(self.path, 'w') as f:
            f.write(self.doc.render())

    def __enter__(self):
        pass

    def __exit__(self, type_, value, tb):
        self.save()


def parse_report(filename, max_frames=None, take_every=None):

    with open(filename, 'r') as f:
        soup = BeautifulSoup(f)

        rows = soup.find_all('table')

        take_every = take_every or 1
        timesteps = range(0, len(rows), take_every)
        rows = [rows[t] for t in timesteps]
        max_frames = max_frames or len(rows)
        rows = rows[:max_frames]
        timesteps = timesteps[:max_frames]

        n_images = len(rows[0].find_all('td'))

        for i in range(n_images):
            title = None
            images = []
            for row in rows:
                td = row.find_all('td')[i]
                if title is None:
                    p = td.find_all('p')[1]
                    title = p.text.split('\n')[-1]
                    print(title)

                img_tag = td.find_all('img')[0]
                _, data = img_tag.attrs['src'].split(',')
                io = BytesIO(b64decode(data))
                img = mpimg.imread(io, format='png')
                images.append(img)

            save_video(title, images, timesteps)


def save_video(title, images, timesteps=None):
    timesteps = timesteps or range(len(images))
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1)
    ax.set_aspect("equal")
    ax.set_axis_off()
    image = ax.imshow(images[0])
    timestep_text = ax.text(
        0.01, 0.01, '', horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)

    def animate(t):
        plt.show()
        image.set_data(images[t])
        timestep_text.set_text("t={}".format(timesteps[t]))

    anim = animation.FuncAnimation(fig, animate, frames=len(images), interval=100)

    gif_filename = title.replace(' ', '_') + '.gif'
    print(gif_filename)
    anim.save(gif_filename, writer='imagemagick')

    plt.close(fig)
