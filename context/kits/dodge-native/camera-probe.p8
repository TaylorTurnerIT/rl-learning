pico-8 cartridge // http://www.pico-8.com
version 42
__lua__
function border(label)
 local value=""
 for x=0,7 do value=value..tostr(pget(x,0)) end
 value=value.."|"
 for y=0,7 do value=value..tostr(pget(0,y)) end
 printh("__camera__"..label.."|"..value)
 local value2=""
 for x=120,127 do value2=value2..tostr(pget(x,127)) end
 printh("__camera2__"..label.."|"..value2)
 cls(0)
 pset(10,10,7)
 local placed=""
 for x=8,12 do placed=placed..tostr(pget(x,10)) end
 printh("__camera3__"..label.."|"..placed)
end

function _draw()
 cls(12)
 border("zero")
 camera(1,0)
 cls(12)
 border("plus1")
 camera(-1,0)
 cls(12)
 border("minus1")
 camera(0,1)
 cls(12)
 border("plusy1")
 camera(0,-1)
 cls(12)
 border("minusy1")
 camera(0.5,0)
 cls(12)
 border("plus05")
 camera(-0.5,0)
 cls(12)
 border("minus05")
 camera(0.5,0.5)
 cls(12)
 border("plusxy05")
 exit()
end
__gfx__
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
