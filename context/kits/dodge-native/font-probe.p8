pico-8 cartridge // http://www.pico-8.com
version 42
__lua__
chars=[[ !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~]]

function _draw()
 cls(0)
 for i=1,#chars do
  print(sub(chars,i,i),((i-1)%16)*8,flr((i-1)/16)*8,7)
 end
 for y=0,127 do
  local row=""
  for x=0,127 do
   row=row..(x==0 and "" or ",")..tostr(pget(x,y))
  end
  printh("__font__"..tostr(y).."|"..row)
 end
 exit()
end
__gfx__
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
