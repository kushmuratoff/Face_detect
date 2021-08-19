from django.db import models
import  time
# Create your models here.
class Person(models.Model):
    Ism         = models.CharField(max_length=150)

    def __str__(self):
        return self.Ism

class Vaqti(models.Model):
    PesonId     = models.ForeignKey(Person, on_delete=models.CASCADE,null=True)
    Input_per    = models.ImageField(upload_to="kunlik/",null=True)
    Output_per    = models.ImageField(upload_to="kunlik/",null=True)
    Vaqti_in    = models.DateTimeField(null=True)
    Vaqti_out   = models.DateTimeField(null=True)

    def timeIn(self):
        return self.Vaqti_in.strftime("%Y.%m.%d %H:%M:%S")

    def timeOut(self):
        return self.Vaqti_out.strftime("%Y.%m.%d %H:%M:%S")

    def A(self):
        return  self.Vaqti_out-self.Vaqti_in







