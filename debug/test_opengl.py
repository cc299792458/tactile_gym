import pyglet
from pyglet import gl
import ctypes

class OpenGLTestWindow(pyglet.window.Window):
    def on_draw(self):
        self.clear()
        # 获取 OpenGL 信息
        version = gl.glGetString(gl.GL_VERSION)
        renderer = gl.glGetString(gl.GL_RENDERER)
        vendor = gl.glGetString(gl.GL_VENDOR)
        
        # 将返回的 LP_c_ubyte 转换为 c_char_p，再 decode() 得到字符串
        version_str = ctypes.cast(version, ctypes.c_char_p).value.decode() if version else "None"
        renderer_str = ctypes.cast(renderer, ctypes.c_char_p).value.decode() if renderer else "None"
        vendor_str = ctypes.cast(vendor, ctypes.c_char_p).value.decode() if vendor else "None"
        
        print("OpenGL 版本:", version_str)
        print("渲染器:", renderer_str)
        print("厂商:", vendor_str)
        pyglet.app.exit()  # 打印后退出

if __name__ == "__main__":
    window = OpenGLTestWindow(width=400, height=300, caption="OpenGL Context Test")
    pyglet.app.run()
