# This file was automatically generated by SWIG (http://www.swig.org).
# Version 3.0.8
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.





from sys import version_info
if version_info >= (2, 6, 0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_cackle', [dirname(__file__)])
        except ImportError:
            import _cackle
            return _cackle
        if fp is not None:
            try:
                _mod = imp.load_module('_cackle', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    _cackle = swig_import_helper()
    del swig_import_helper
else:
    import _cackle
del version_info
try:
    _swig_property = property
except NameError:
    pass  # Python < 2.2 doesn't have 'property'.


def _swig_setattr_nondynamic(self, class_type, name, value, static=1):
    if (name == "thisown"):
        return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name, None)
    if method:
        return method(self, value)
    if (not static):
        if _newclass:
            object.__setattr__(self, name, value)
        else:
            self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)


def _swig_setattr(self, class_type, name, value):
    return _swig_setattr_nondynamic(self, class_type, name, value, 0)


def _swig_getattr_nondynamic(self, class_type, name, static=1):
    if (name == "thisown"):
        return self.this.own()
    method = class_type.__swig_getmethods__.get(name, None)
    if method:
        return method(self)
    if (not static):
        return object.__getattr__(self, name)
    else:
        raise AttributeError(name)

def _swig_getattr(self, class_type, name):
    return _swig_getattr_nondynamic(self, class_type, name, 0)


def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except AttributeError:
    class _object:
        pass
    _newclass = 0


class float4(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, float4, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, float4, name)
    __repr__ = _swig_repr

    def __init__(self, *args):
        this = _cackle.new_float4(*args)
        try:
            self.this.append(this)
        except Exception:
            self.this = this
    __swig_setmethods__["x"] = _cackle.float4_x_set
    __swig_getmethods__["x"] = _cackle.float4_x_get
    if _newclass:
        x = _swig_property(_cackle.float4_x_get, _cackle.float4_x_set)
    __swig_setmethods__["y"] = _cackle.float4_y_set
    __swig_getmethods__["y"] = _cackle.float4_y_get
    if _newclass:
        y = _swig_property(_cackle.float4_y_get, _cackle.float4_y_set)
    __swig_setmethods__["z"] = _cackle.float4_z_set
    __swig_getmethods__["z"] = _cackle.float4_z_get
    if _newclass:
        z = _swig_property(_cackle.float4_z_get, _cackle.float4_z_set)
    __swig_setmethods__["w"] = _cackle.float4_w_set
    __swig_getmethods__["w"] = _cackle.float4_w_get
    if _newclass:
        w = _swig_property(_cackle.float4_w_get, _cackle.float4_w_set)
    __swig_destroy__ = _cackle.delete_float4
    __del__ = lambda self: None
float4_swigregister = _cackle.float4_swigregister
float4_swigregister(float4)

class Camera(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Camera, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Camera, name)
    __repr__ = _swig_repr

    def __init__(self, *args):
        this = _cackle.new_Camera(*args)
        try:
            self.this.append(this)
        except Exception:
            self.this = this
    __swig_setmethods__["eye"] = _cackle.Camera_eye_set
    __swig_getmethods__["eye"] = _cackle.Camera_eye_get
    if _newclass:
        eye = _swig_property(_cackle.Camera_eye_get, _cackle.Camera_eye_set)
    __swig_setmethods__["aim"] = _cackle.Camera_aim_set
    __swig_getmethods__["aim"] = _cackle.Camera_aim_get
    if _newclass:
        aim = _swig_property(_cackle.Camera_aim_get, _cackle.Camera_aim_set)
    __swig_setmethods__["view"] = _cackle.Camera_view_set
    __swig_getmethods__["view"] = _cackle.Camera_view_get
    if _newclass:
        view = _swig_property(_cackle.Camera_view_get, _cackle.Camera_view_set)
    __swig_setmethods__["up"] = _cackle.Camera_up_set
    __swig_getmethods__["up"] = _cackle.Camera_up_get
    if _newclass:
        up = _swig_property(_cackle.Camera_up_get, _cackle.Camera_up_set)
    __swig_setmethods__["right"] = _cackle.Camera_right_set
    __swig_getmethods__["right"] = _cackle.Camera_right_get
    if _newclass:
        right = _swig_property(_cackle.Camera_right_get, _cackle.Camera_right_set)
    __swig_setmethods__["fov"] = _cackle.Camera_fov_set
    __swig_getmethods__["fov"] = _cackle.Camera_fov_get
    if _newclass:
        fov = _swig_property(_cackle.Camera_fov_get, _cackle.Camera_fov_set)
    __swig_destroy__ = _cackle.delete_Camera
    __del__ = lambda self: None
Camera_swigregister = _cackle.Camera_swigregister
Camera_swigregister(Camera)

class HDRPointLight(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, HDRPointLight, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, HDRPointLight, name)
    __repr__ = _swig_repr

    def __init__(self, *args):
        this = _cackle.new_HDRPointLight(*args)
        try:
            self.this.append(this)
        except Exception:
            self.this = this
    __swig_setmethods__["p"] = _cackle.HDRPointLight_p_set
    __swig_getmethods__["p"] = _cackle.HDRPointLight_p_get
    if _newclass:
        p = _swig_property(_cackle.HDRPointLight_p_get, _cackle.HDRPointLight_p_set)
    __swig_setmethods__["color"] = _cackle.HDRPointLight_color_set
    __swig_getmethods__["color"] = _cackle.HDRPointLight_color_get
    if _newclass:
        color = _swig_property(_cackle.HDRPointLight_color_get, _cackle.HDRPointLight_color_set)
    __swig_setmethods__["falloff"] = _cackle.HDRPointLight_falloff_set
    __swig_getmethods__["falloff"] = _cackle.HDRPointLight_falloff_get
    if _newclass:
        falloff = _swig_property(_cackle.HDRPointLight_falloff_get, _cackle.HDRPointLight_falloff_set)
    __swig_setmethods__["px"] = _cackle.HDRPointLight_px_set
    __swig_getmethods__["px"] = _cackle.HDRPointLight_px_get
    if _newclass:
        px = _swig_property(_cackle.HDRPointLight_px_get, _cackle.HDRPointLight_px_set)
    __swig_setmethods__["py"] = _cackle.HDRPointLight_py_set
    __swig_getmethods__["py"] = _cackle.HDRPointLight_py_get
    if _newclass:
        py = _swig_property(_cackle.HDRPointLight_py_get, _cackle.HDRPointLight_py_set)
    __swig_destroy__ = _cackle.delete_HDRPointLight
    __del__ = lambda self: None
HDRPointLight_swigregister = _cackle.HDRPointLight_swigregister
HDRPointLight_swigregister(HDRPointLight)

class Uniforms(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, Uniforms, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, Uniforms, name)
    __repr__ = _swig_repr

    def __init__(self, *args):
        this = _cackle.new_Uniforms(*args)
        try:
            self.this.append(this)
        except Exception:
            self.this = this
    __swig_setmethods__["use_path_tracing"] = _cackle.Uniforms_use_path_tracing_set
    __swig_getmethods__["use_path_tracing"] = _cackle.Uniforms_use_path_tracing_get
    if _newclass:
        use_path_tracing = _swig_property(_cackle.Uniforms_use_path_tracing_get, _cackle.Uniforms_use_path_tracing_set)
    __swig_setmethods__["max_path_length"] = _cackle.Uniforms_max_path_length_set
    __swig_getmethods__["max_path_length"] = _cackle.Uniforms_max_path_length_get
    if _newclass:
        max_path_length = _swig_property(_cackle.Uniforms_max_path_length_get, _cackle.Uniforms_max_path_length_set)
    __swig_setmethods__["samples"] = _cackle.Uniforms_samples_set
    __swig_getmethods__["samples"] = _cackle.Uniforms_samples_get
    if _newclass:
        samples = _swig_property(_cackle.Uniforms_samples_get, _cackle.Uniforms_samples_set)
    __swig_setmethods__["dither"] = _cackle.Uniforms_dither_set
    __swig_getmethods__["dither"] = _cackle.Uniforms_dither_get
    if _newclass:
        dither = _swig_property(_cackle.Uniforms_dither_get, _cackle.Uniforms_dither_set)
    __swig_setmethods__["fudgefactor"] = _cackle.Uniforms_fudgefactor_set
    __swig_getmethods__["fudgefactor"] = _cackle.Uniforms_fudgefactor_get
    if _newclass:
        fudgefactor = _swig_property(_cackle.Uniforms_fudgefactor_get, _cackle.Uniforms_fudgefactor_set)
    __swig_setmethods__["fractal_iterations"] = _cackle.Uniforms_fractal_iterations_set
    __swig_getmethods__["fractal_iterations"] = _cackle.Uniforms_fractal_iterations_get
    if _newclass:
        fractal_iterations = _swig_property(_cackle.Uniforms_fractal_iterations_get, _cackle.Uniforms_fractal_iterations_set)
    __swig_setmethods__["raymarch_iterations"] = _cackle.Uniforms_raymarch_iterations_set
    __swig_getmethods__["raymarch_iterations"] = _cackle.Uniforms_raymarch_iterations_get
    if _newclass:
        raymarch_iterations = _swig_property(_cackle.Uniforms_raymarch_iterations_get, _cackle.Uniforms_raymarch_iterations_set)
    __swig_setmethods__["flimit"] = _cackle.Uniforms_flimit_set
    __swig_getmethods__["flimit"] = _cackle.Uniforms_flimit_get
    if _newclass:
        flimit = _swig_property(_cackle.Uniforms_flimit_get, _cackle.Uniforms_flimit_set)
    __swig_setmethods__["epsilon"] = _cackle.Uniforms_epsilon_set
    __swig_getmethods__["epsilon"] = _cackle.Uniforms_epsilon_get
    if _newclass:
        epsilon = _swig_property(_cackle.Uniforms_epsilon_get, _cackle.Uniforms_epsilon_set)
    __swig_setmethods__["scale"] = _cackle.Uniforms_scale_set
    __swig_getmethods__["scale"] = _cackle.Uniforms_scale_get
    if _newclass:
        scale = _swig_property(_cackle.Uniforms_scale_get, _cackle.Uniforms_scale_set)
    __swig_setmethods__["mr2"] = _cackle.Uniforms_mr2_set
    __swig_getmethods__["mr2"] = _cackle.Uniforms_mr2_get
    if _newclass:
        mr2 = _swig_property(_cackle.Uniforms_mr2_get, _cackle.Uniforms_mr2_set)
    __swig_setmethods__["fr2"] = _cackle.Uniforms_fr2_set
    __swig_getmethods__["fr2"] = _cackle.Uniforms_fr2_get
    if _newclass:
        fr2 = _swig_property(_cackle.Uniforms_fr2_get, _cackle.Uniforms_fr2_set)
    __swig_setmethods__["julia_c"] = _cackle.Uniforms_julia_c_set
    __swig_getmethods__["julia_c"] = _cackle.Uniforms_julia_c_get
    if _newclass:
        julia_c = _swig_property(_cackle.Uniforms_julia_c_get, _cackle.Uniforms_julia_c_set)
    __swig_setmethods__["julia_offset"] = _cackle.Uniforms_julia_offset_set
    __swig_getmethods__["julia_offset"] = _cackle.Uniforms_julia_offset_get
    if _newclass:
        julia_offset = _swig_property(_cackle.Uniforms_julia_offset_get, _cackle.Uniforms_julia_offset_set)
    __swig_setmethods__["de_offset"] = _cackle.Uniforms_de_offset_set
    __swig_getmethods__["de_offset"] = _cackle.Uniforms_de_offset_get
    if _newclass:
        de_offset = _swig_property(_cackle.Uniforms_de_offset_get, _cackle.Uniforms_de_offset_set)
    __swig_setmethods__["fog_level"] = _cackle.Uniforms_fog_level_set
    __swig_getmethods__["fog_level"] = _cackle.Uniforms_fog_level_get
    if _newclass:
        fog_level = _swig_property(_cackle.Uniforms_fog_level_get, _cackle.Uniforms_fog_level_set)
    __swig_setmethods__["fog_color"] = _cackle.Uniforms_fog_color_set
    __swig_getmethods__["fog_color"] = _cackle.Uniforms_fog_color_get
    if _newclass:
        fog_color = _swig_property(_cackle.Uniforms_fog_color_get, _cackle.Uniforms_fog_color_set)
    __swig_setmethods__["glow_strength"] = _cackle.Uniforms_glow_strength_set
    __swig_getmethods__["glow_strength"] = _cackle.Uniforms_glow_strength_get
    if _newclass:
        glow_strength = _swig_property(_cackle.Uniforms_glow_strength_get, _cackle.Uniforms_glow_strength_set)
    __swig_setmethods__["u_time"] = _cackle.Uniforms_u_time_set
    __swig_getmethods__["u_time"] = _cackle.Uniforms_u_time_get
    if _newclass:
        u_time = _swig_property(_cackle.Uniforms_u_time_get, _cackle.Uniforms_u_time_set)
    __swig_setmethods__["scalefactor"] = _cackle.Uniforms_scalefactor_set
    __swig_getmethods__["scalefactor"] = _cackle.Uniforms_scalefactor_get
    if _newclass:
        scalefactor = _swig_property(_cackle.Uniforms_scalefactor_get, _cackle.Uniforms_scalefactor_set)
    __swig_setmethods__["inverse_scalefactor"] = _cackle.Uniforms_inverse_scalefactor_set
    __swig_getmethods__["inverse_scalefactor"] = _cackle.Uniforms_inverse_scalefactor_get
    if _newclass:
        inverse_scalefactor = _swig_property(_cackle.Uniforms_inverse_scalefactor_get, _cackle.Uniforms_inverse_scalefactor_set)
    __swig_setmethods__["bv_radius"] = _cackle.Uniforms_bv_radius_set
    __swig_getmethods__["bv_radius"] = _cackle.Uniforms_bv_radius_get
    if _newclass:
        bv_radius = _swig_property(_cackle.Uniforms_bv_radius_get, _cackle.Uniforms_bv_radius_set)
    __swig_setmethods__["specular_exponent"] = _cackle.Uniforms_specular_exponent_set
    __swig_getmethods__["specular_exponent"] = _cackle.Uniforms_specular_exponent_get
    if _newclass:
        specular_exponent = _swig_property(_cackle.Uniforms_specular_exponent_get, _cackle.Uniforms_specular_exponent_set)
    __swig_setmethods__["gamma"] = _cackle.Uniforms_gamma_set
    __swig_getmethods__["gamma"] = _cackle.Uniforms_gamma_get
    if _newclass:
        gamma = _swig_property(_cackle.Uniforms_gamma_get, _cackle.Uniforms_gamma_set)
    __swig_setmethods__["reflectivity"] = _cackle.Uniforms_reflectivity_set
    __swig_getmethods__["reflectivity"] = _cackle.Uniforms_reflectivity_get
    if _newclass:
        reflectivity = _swig_property(_cackle.Uniforms_reflectivity_get, _cackle.Uniforms_reflectivity_set)
    __swig_setmethods__["albedo"] = _cackle.Uniforms_albedo_set
    __swig_getmethods__["albedo"] = _cackle.Uniforms_albedo_get
    if _newclass:
        albedo = _swig_property(_cackle.Uniforms_albedo_get, _cackle.Uniforms_albedo_set)
    __swig_setmethods__["colormap_a"] = _cackle.Uniforms_colormap_a_set
    __swig_getmethods__["colormap_a"] = _cackle.Uniforms_colormap_a_get
    if _newclass:
        colormap_a = _swig_property(_cackle.Uniforms_colormap_a_get, _cackle.Uniforms_colormap_a_set)
    __swig_setmethods__["colormap_b"] = _cackle.Uniforms_colormap_b_set
    __swig_getmethods__["colormap_b"] = _cackle.Uniforms_colormap_b_get
    if _newclass:
        colormap_b = _swig_property(_cackle.Uniforms_colormap_b_get, _cackle.Uniforms_colormap_b_set)
    __swig_setmethods__["colormap_c"] = _cackle.Uniforms_colormap_c_set
    __swig_getmethods__["colormap_c"] = _cackle.Uniforms_colormap_c_get
    if _newclass:
        colormap_c = _swig_property(_cackle.Uniforms_colormap_c_get, _cackle.Uniforms_colormap_c_set)
    __swig_setmethods__["colormap_d"] = _cackle.Uniforms_colormap_d_set
    __swig_getmethods__["colormap_d"] = _cackle.Uniforms_colormap_d_get
    if _newclass:
        colormap_d = _swig_property(_cackle.Uniforms_colormap_d_get, _cackle.Uniforms_colormap_d_set)
    __swig_setmethods__["hdrmap_width"] = _cackle.Uniforms_hdrmap_width_set
    __swig_getmethods__["hdrmap_width"] = _cackle.Uniforms_hdrmap_width_get
    if _newclass:
        hdrmap_width = _swig_property(_cackle.Uniforms_hdrmap_width_get, _cackle.Uniforms_hdrmap_width_set)
    __swig_setmethods__["hdrmap_height"] = _cackle.Uniforms_hdrmap_height_set
    __swig_getmethods__["hdrmap_height"] = _cackle.Uniforms_hdrmap_height_get
    if _newclass:
        hdrmap_height = _swig_property(_cackle.Uniforms_hdrmap_height_get, _cackle.Uniforms_hdrmap_height_set)
    __swig_setmethods__["hdrmap_rotation"] = _cackle.Uniforms_hdrmap_rotation_set
    __swig_getmethods__["hdrmap_rotation"] = _cackle.Uniforms_hdrmap_rotation_get
    if _newclass:
        hdrmap_rotation = _swig_property(_cackle.Uniforms_hdrmap_rotation_get, _cackle.Uniforms_hdrmap_rotation_set)
    __swig_setmethods__["hdrlight_count"] = _cackle.Uniforms_hdrlight_count_set
    __swig_getmethods__["hdrlight_count"] = _cackle.Uniforms_hdrlight_count_get
    if _newclass:
        hdrlight_count = _swig_property(_cackle.Uniforms_hdrlight_count_get, _cackle.Uniforms_hdrlight_count_set)
    __swig_destroy__ = _cackle.delete_Uniforms
    __del__ = lambda self: None
Uniforms_swigregister = _cackle.Uniforms_swigregister
Uniforms_swigregister(Uniforms)

class CackleRenderer(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, CackleRenderer, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, CackleRenderer, name)
    __repr__ = _swig_repr

    def __init__(self, *args):
        this = _cackle.new_CackleRenderer(*args)
        try:
            self.this.append(this)
        except Exception:
            self.this = this
    __swig_destroy__ = _cackle.delete_CackleRenderer
    __del__ = lambda self: None

    def init(self):
        return _cackle.CackleRenderer_init(self)

    def initCubemap(self):
        return _cackle.CackleRenderer_initCubemap(self)

    def render(self):
        return _cackle.CackleRenderer_render(self)

    def renderChunk(self, cx, n_chunks):
        return _cackle.CackleRenderer_renderChunk(self, cx, n_chunks)

    def renderCubemap(self):
        return _cackle.CackleRenderer_renderCubemap(self)

    def outputImage(self, filepath):
        return _cackle.CackleRenderer_outputImage(self, filepath)

    def outputCubemap(self, filepath):
        return _cackle.CackleRenderer_outputCubemap(self, filepath)

    def setCameraEye(self, x, y, z):
        return _cackle.CackleRenderer_setCameraEye(self, x, y, z)

    def setCameraAim(self, x, y, z):
        return _cackle.CackleRenderer_setCameraAim(self, x, y, z)

    def setCameraView(self, x, y, z):
        return _cackle.CackleRenderer_setCameraView(self, x, y, z)

    def setCameraUp(self, x, y, z):
        return _cackle.CackleRenderer_setCameraUp(self, x, y, z)

    def setCameraRight(self, x, y, z):
        return _cackle.CackleRenderer_setCameraRight(self, x, y, z)

    def setCameraFOV(self, fov):
        return _cackle.CackleRenderer_setCameraFOV(self, fov)

    def usePathTracing(self, upt):
        return _cackle.CackleRenderer_usePathTracing(self, upt)

    def setMaxPathLength(self, mpl):
        return _cackle.CackleRenderer_setMaxPathLength(self, mpl)

    def setSampleCount(self, sc):
        return _cackle.CackleRenderer_setSampleCount(self, sc)

    def setDither(self, dither):
        return _cackle.CackleRenderer_setDither(self, dither)

    def setFudgeFactor(self, fudgefactor):
        return _cackle.CackleRenderer_setFudgeFactor(self, fudgefactor)

    def setFractalIterations(self, fi):
        return _cackle.CackleRenderer_setFractalIterations(self, fi)

    def setRaymarchIterations(self, ri):
        return _cackle.CackleRenderer_setRaymarchIterations(self, ri)

    def setFLimit(self, x, y, z):
        return _cackle.CackleRenderer_setFLimit(self, x, y, z)

    def setJuliaC(self, x, y, z):
        return _cackle.CackleRenderer_setJuliaC(self, x, y, z)

    def setJuliaOffset(self, x, y, z):
        return _cackle.CackleRenderer_setJuliaOffset(self, x, y, z)

    def setDEOffset(self, deo):
        return _cackle.CackleRenderer_setDEOffset(self, deo)

    def setEpsilon(self, eps):
        return _cackle.CackleRenderer_setEpsilon(self, eps)

    def setScale(self, scale):
        return _cackle.CackleRenderer_setScale(self, scale)

    def setMR2(self, mr2):
        return _cackle.CackleRenderer_setMR2(self, mr2)

    def setFR2(self, fr2):
        return _cackle.CackleRenderer_setFR2(self, fr2)

    def setFogLevel(self, fog):
        return _cackle.CackleRenderer_setFogLevel(self, fog)

    def setFogColor(self, r, g, b):
        return _cackle.CackleRenderer_setFogColor(self, r, g, b)

    def setGlowStrength(self, glow):
        return _cackle.CackleRenderer_setGlowStrength(self, glow)

    def setSpecularExponent(self, spec):
        return _cackle.CackleRenderer_setSpecularExponent(self, spec)

    def setGamma(self, g):
        return _cackle.CackleRenderer_setGamma(self, g)

    def setReflectivity(self, refl):
        return _cackle.CackleRenderer_setReflectivity(self, refl)

    def setAlbedo(self, a):
        return _cackle.CackleRenderer_setAlbedo(self, a)

    def addPointLight(self, x, y, z, r, g, b, falloff):
        return _cackle.CackleRenderer_addPointLight(self, x, y, z, r, g, b, falloff)

    def setColormapA(self, r, g, b):
        return _cackle.CackleRenderer_setColormapA(self, r, g, b)

    def setColormapB(self, r, g, b):
        return _cackle.CackleRenderer_setColormapB(self, r, g, b)

    def setColormapC(self, r, g, b):
        return _cackle.CackleRenderer_setColormapC(self, r, g, b)

    def setColormapD(self, r, g, b):
        return _cackle.CackleRenderer_setColormapD(self, r, g, b)

    def setColormap(self, id):
        return _cackle.CackleRenderer_setColormap(self, id)

    def setHDRMapRotation(self, mr):
        return _cackle.CackleRenderer_setHDRMapRotation(self, mr)

    def setHDRLightCount(self, lc):
        return _cackle.CackleRenderer_setHDRLightCount(self, lc)

    def setHDRMap(self, path):
        return _cackle.CackleRenderer_setHDRMap(self, path)
CackleRenderer_swigregister = _cackle.CackleRenderer_swigregister
CackleRenderer_swigregister(CackleRenderer)

# This file is compatible with both classic and new-style classes.


