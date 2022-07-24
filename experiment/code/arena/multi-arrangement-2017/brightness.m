function b=brightness(RGBrows)

RGBweights=[.241 .691 .068]';

b=sqrt(RGBrows*RGBweights);

