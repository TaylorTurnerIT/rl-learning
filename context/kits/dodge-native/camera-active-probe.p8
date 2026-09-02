pico-8 cartridge // http://www.pico-8.com
version 42
__lua__
function sample(label,cx,cy)
 camera(0,0)
 cls(12)
 camera(cx,cy)
 pset(10,10,7)
 local values=""
 for y=124,127 do
  values=values..(values!="" and ";" or "")..tostr(y)..","..tostr(pget(10,y))
 end
 printh("__camera_active__"..label.."|"..values)
end

function _draw()
 sample("zero",0,0)
 sample("minus05",0,-0.5)
 sample("minus1",0,-1)
 sample("minus13",0,-1.3)
 sample("plus05",0,0.5)
 sample("plus1",0,1)
 camera(0,0)
 exit()
end
__gfx__
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
