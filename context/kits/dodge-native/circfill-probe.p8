pico-8 cartridge // http://www.pico-8.com
version 42
__lua__
function probe_float(radius,label)
 cls(12)
 circfill(64,64,radius,7)
 for y=58,70 do
  local bits=""
  for x=58,70 do
   bits=bits..(pget(x,y)==7 and "1" or "0")
  end
  printh("__circlefloat__"..label.."|"..(y-64).."|"..bits)
 end
end

function _draw()
 for radius=0,8 do
  cls(0)
  circfill(32,32,radius,7)
  for y=32-radius-1,32+radius+1 do
   local bits=""
   for x=32-radius-1,32+radius+1 do
    bits=bits..(pget(x,y)==7 and "1" or "0")
   end
   printh("__circle__"..radius.."|"..(y-32).."|"..bits)
  end
 end
 cls(12)
 circfill(64,64,4,7)
 for y=56,72 do
  local bits=""
  for x=56,72 do
   bits=bits..(pget(x,y)==7 and "1" or "0")
  end
  printh("__circle64__"..(y-64).."|"..bits)
 end
 probe_float(2.9159,"29159")
 probe_float(3.2399,"32399")
 probe_float(3.5999,"35999")
 probe_float(4,"4")
 exit()
end
__gfx__
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
