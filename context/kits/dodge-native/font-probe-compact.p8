pico-8 cartridge // http://www.pico-8.com
version 42
__lua__
chars=[[ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz 0123456789:;,.!?+-><()]]

function emitglyph(i,x,y)
  for row=0,7 do
  local bits=""
  for col=0,2 do
   bits=bits..(pget(x+col,y+row)==7 and "1" or "0")
  end
  printh("__glyph__"..i.."|"..row.."|"..bits)
 end
end

function _draw()
 cls(0)
 for i=1,#chars do
  local x=((i-1)%16)*8
  local y=flr((i-1)/16)*8
  print(sub(chars,i,i),x,y,7)
  emitglyph(i,x,y)
 end
 exit()
end
__gfx__
00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000
