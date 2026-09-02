pico-8 cartridge // http://www.pico-8.com
version 42
__lua__
function signature()
 local value=""
 for dy=-6,6 do
  local count=0
  for x=50,78 do
   if pget(x,64+dy)==7 then count+=1 end
  end
  value=value..(dy==-6 and "" or ",")..tostr(count)
 end
 return value
end

function _draw()
 local previous=""
 for i=0,400 do
  local radius=i/100
  cls(0)
  circfill(64,64,radius,7)
  local current=signature()
  if current!=previous then
   printh("__threshold__"..tostr(i).."|"..current)
   previous=current
  end
 end
 exit()
end
__gfx__
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
