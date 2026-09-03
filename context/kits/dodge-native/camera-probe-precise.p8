pico-8 cartridge // http://www.pico-8.com
version 42
__lua__
function sample(label,cx,cy)
 camera(0,0)
 cls(12)
 camera(cx,cy)
 pset(10,10,7)
 rectfill(12,10,13,11,8)
 print("0",1,1,7)
 camera(0,0)
 local values={}
 for y=0,20 do
  for x=0,20 do
   local value=pget(x,y)
   if value!=12 then add(values,tostr(x)..","..tostr(y)..","..tostr(value)) end
  end
 end
 printh("__camera_precise__"..label.."|"..__join(values))
end

function __join(values)
 local result=""
 for value in all(values) do
  result=result..(result!="" and ";" or "")..value
 end
 return result
end

function _draw()
 sample("zero",0,0)
 sample("plus1",1,0)
 sample("minus1",-1,0)
 sample("plus05",0.5,0)
 sample("minus05",-0.5,0)
 sample("plus13",1.348,0)
 sample("plusy1",0,1)
 sample("minusy1",0,-1)
 sample("minus13",0,-1.319)
 camera(0,0)
 exit()
end
__gfx__
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
