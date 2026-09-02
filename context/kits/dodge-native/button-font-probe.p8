pico-8 cartridge // http://www.pico-8.com
version 42
__lua__
chars={"❎","🅾️","⬅️","➡️","⬆️","⬇️"}

function _draw()
 cls(0)
 for i=1,#chars do
  local x=(i-1)*8
  print(chars[i],x,0,7)
  for row=0,7 do
   local bits=""
   for col=0,7 do
    bits=bits..(pget(x+col,row)==7 and "1" or "0")
   end
   printh("__special__"..i.."|"..row.."|"..bits)
  end
 end
 exit()
end
__gfx__
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
